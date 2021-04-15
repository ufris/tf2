import math, csv
import tensorflow as tf
import tensorflow_addons as tfa
import random
from datetime import datetime
import imutils, os, cv2
import numpy as np
from tf2.image_util import *
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
input_size = 331
save_ckpt_path = '/media/crescom/새 볼륨/ckpt/AS/' + '/'
category = 'tf2_L_spine_UP_rotation_sampling_cut_3000_shuffle_ensemble_resnet_add_y_x_fifth_fix_label_smooth'
epoch = 100
total_path = '/media/crescom/새 볼륨/data/0.1_final_up_down_L/fifth_fix/up'
mini_batch_size = 3
binary_train = False
initial_learning_rate = 0.000001
metric = 'mae'

print(tf.__version__)

gpus = tf.config.experimental.list_physical_devices('GPU')
print('gpus',gpus)
if gpus:
    try:
        # Restrict TensorFlow to only use the fourth GPU
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

def for_directory(direc, cnt):
    if not os.path.exists(direc):
        os.makedirs(direc)
        os.chdir(direc)
    elif os.path.exists(direc):
        cnt += 1
        direc = direc + '_copy'
        for_directory(direc, cnt)

def save_parameter():
    f = open('./parameter' + '.txt', 'w')
    f.write('img_size : ' + str(input_size) + '\n')
    f.write('category : ' + str(category) + '\n')
    f.write('mini_batch_size : ' + str(mini_batch_size) + '\n')
    f.close()

i = datetime.now()
current_date = i.strftime('%Y/%m/%d').replace('/', '_')
summary_name = save_ckpt_path + current_date
cnt = 0

def load_class_X_Y(img_path, set_name):
    top_img_path = img_path + '/' + set_name + '/'
    class_name = sorted(os.listdir(top_img_path))

    img_list = []
    label_list = []

    for one_class in range(len(class_name)):
        one_class_path = top_img_path + class_name[one_class] + '/'
        one_class_img_name = os.listdir(one_class_path)
        if class_name[one_class] == '0':
            shuffle_int = random.randint(0,100)
            random.seed(shuffle_int)
            random.shuffle(one_class_img_name)
            for one_class_one_img_name in one_class_img_name[:3000]:
                img_list.append(one_class_path + one_class_one_img_name)
                label_list.append(one_class)
        else:
            for one_class_one_img_name in one_class_img_name:
                img_list.append(one_class_path + one_class_one_img_name)
                label_list.append(one_class)

    return img_list, label_list


def one_hot_Y(train_Y_list, class_cnt):
    train_Y = np.eye(class_cnt)[train_Y_list]

    return train_Y

def random_shuffle(x, seed):
    random.seed(seed)
    random.shuffle(x)
    return x

train_img_list, train_label_list = load_class_X_Y(total_path, 'train')
val_img_list, val_label_list = load_class_X_Y(total_path, 'val')

output_cnt = len(set(train_label_list))

# train_img_list, train_label_list = train_img_list[:10], train_label_list[:10]
# val_img_list, val_label_list = val_img_list[:10], val_label_list[:10]

print(train_img_list[:10], train_label_list[:10])
# print(val_img_list[:-10], val_label_list[:-10])

# 슬라이스를 이용한 next batch
def next_batch(data_list, mini_batch_size, next_cnt):
    cnt = mini_batch_size * next_cnt
    batch_list = data_list[cnt:cnt + mini_batch_size]
    return batch_list

val_mini_batch_size = 1
batch_size = math.ceil(len(train_img_list) / mini_batch_size)
val_batch_size = math.ceil(len(val_img_list) / val_mini_batch_size)
print(batch_size)
batch_cnt = 0
val_batch_cnt = 0
max_val_acc = 0

def img_loader(img_list, rot_rate=0, shift_rate=0.0, aug=''):
    img_set = []
    for one_img_list in img_list:
        one_img = cv2.imread(one_img_list)

        if aug == 'train':
            img_preprocessing = random.randint(0,2)
            one_img = clahe(one_img) if img_preprocessing == 0 else normalize(one_img) if img_preprocessing == 1 else one_img

            rot = random.randint(0, rot_rate)

            x, y = one_img.shape[1], one_img.shape[0]

            shift_x = random.randint(-int(x * shift_rate), int(x * shift_rate))
            shift_y = random.randint(-int(y * shift_rate), int(y * shift_rate))

            # filp_ran = random.randint(0, 1)
            # if filp_ran:
            #     # vertical flip
            #     one_img = cv2.flip(one_img, 1)

            one_img = imutils.translate(one_img, shift_x, shift_y)
            one_img = imutils.rotate(one_img, rot)
        elif aug == 'ensemble_test':
            clahe_img = cv2.resize(clahe(one_img), (input_size, input_size))
            normal_img = cv2.resize(normalize(one_img), (input_size, input_size))
            one_img = cv2.resize(one_img, (input_size, input_size))

            img_set = np.stack([one_img,clahe_img,normal_img], axis=0)
            img_set = img_set.astype(np.float32)
            img_set /= 255.0

            return img_set

        one_img = cv2.resize(one_img, (input_size, input_size))

        img_set.append(one_img)

    img_set = np.array(img_set, dtype=np.float32)

    img_set /= 255.0

    return img_set

l2_regul = tf.keras.regularizers.l2(l=0.1)
he_init = tf.keras.initializers.he_normal()

IMG_SHAPE = (input_size,input_size,3)

base_model = tf.keras.applications.ResNet152V2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet',)

base_model.trainable = True

# rescale = tf.keras.applications.nasnet.preprocess_input
# model_output = base_model.output
#
# global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
# prediction_layer = tf.keras.layers.Dense(4, activation='softmax',kernel_initializer=he_init)

# model = tf.keras.Sequential([
#     base_model,
#     global_average_layer,
#     prediction_layer
# ])

x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
if binary_train:
    outputs = tf.keras.layers.Dense(1, kernel_initializer=he_init)(x)
else:
    outputs = tf.keras.layers.Dense(4, activation='softmax',kernel_initializer=he_init)(x)
model = tf.keras.Model(inputs=base_model.input,outputs=outputs)

# tf.keras.models.load_model('/media/crescom2/DATA/temp')
print('load model')

model.summary()

# decay_steps = 1000
# lr_decayed_fn = tf.keras.experimental.CosineDecay(initial_learning_rate, decay_steps)
optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)
model_vars = model.trainable_variables
ema = tf.train.ExponentialMovingAverage(decay=0.999)
with tf.control_dependencies([optimizer]):
    train_op = ema.apply(model_vars)

# ema = tf.train.ExponentialMovingAverage(0.999)
# with tf.control_dependencies([optimizer]):
#     training_op = ema.apply([var0, var1])
# tfa.optimizers.moving_average(optimizer)

# model.load_weights('/media/crescom/새 볼륨1/ckpt/spine_cls/2020_10_08_copy/nasnet_fine_tuned_model.h5')

train_accuracy_list = []
train_loss_list = []
val_accuracy_list = []

best_accuracy = 0

print(summary_name)
for_directory(summary_name, cnt)
save_parameter()

for k in range(epoch):
    print('epoch : ', k)
    val_accuracy = 0
    train_accuracy = 0
    train_loss = 0

    ranint = random.randint(1, 100)
    train_img_list = random_shuffle(train_img_list, ranint)
    train_label_list = random_shuffle(train_label_list, ranint)

    for i in range(batch_size):
        train_one_batch_X_list = next_batch(train_img_list, mini_batch_size, i)
        train_one_batch_Y = next_batch(train_label_list, mini_batch_size, i)

        if not binary_train:
            train_one_batch_Y = one_hot_Y(train_one_batch_Y, 4)
        train_one_batch_X = img_loader(train_one_batch_X_list, 15, 0.1, 'train')

        with tf.GradientTape() as tape:
            y_ = model(train_one_batch_X)

            # round(model.optimizer.lr.numpy(), 5)

            if binary_train:
                loss = tf.keras.losses.mean_squared_error(y_true=train_one_batch_Y, y_pred=y_)
            else:
                loss = tf.keras.losses.categorical_crossentropy(y_true=train_one_batch_Y, y_pred=y_, label_smoothing=0.15)
                # loss = tf.keras.losses.CategoricalCrossentropy()(y_true=train_one_batch_Y, y_pred=y_)
        print('train_one_batch_Y',train_one_batch_Y)
        grad = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grad, model.trainable_variables))

        if binary_train:
            MSE = np.square(np.subtract(train_one_batch_Y,y_)).mean()
            batch_loss = np.mean(loss)

            train_accuracy += MSE
            train_loss += batch_loss

            print('train loss :', batch_loss)
            print('train acc :', MSE)
        else:
            batch_acc = np.mean(np.cast[np.int32](np.equal(np.argmax(y_,axis=1), np.argmax(train_one_batch_Y,axis=1))))
            batch_loss = np.mean(loss)

            train_accuracy += batch_acc
            train_loss += batch_loss

            print('train loss :',batch_loss)
            print('train acc :',batch_acc)

    train_accuracy = round(train_accuracy / batch_size,5)
    train_loss = round(train_loss / batch_size,5)

    train_accuracy_list.append(train_accuracy)
    train_loss_list.append(train_loss)
    print('train_accuracy :',train_accuracy)
    print('train_loss :',train_loss)

    # print('batch : ', batch_cnt, 'train_acc : ', one_batch_train_acc, 'train loss : ', loss)

    for i in range(val_batch_size):
        val_one_batch_X_list = next_batch(val_img_list, val_mini_batch_size, i)
        val_one_batch_Y = next_batch(val_label_list, val_mini_batch_size, i)

        if not binary_train:
            val_one_batch_Y = one_hot_Y(val_one_batch_Y, output_cnt)
        val_one_batch_X = img_loader(val_one_batch_X_list, 0, 0, aug='ensemble_test')

        y_ = model(val_one_batch_X)

        if binary_train:
            class_preds_sum = np.sum(y_, axis=0) / 3

            val_MSE = np.square(np.subtract(val_one_batch_Y, class_preds_sum)).mean()

            val_accuracy += val_MSE

        else:
            class_idx = np.argmax(y_, axis=1)
            class_preds_sum = np.sum(y_, axis=0) / 3

            ensemble_predict = np.argmax(class_preds_sum)
            y = np.argmax(val_one_batch_Y)

            if metric == 'acc':
                val_acc = np.mean(np.cast[np.int32](np.equal(y, ensemble_predict)))
            elif metric == 'mae':
                val_acc = tf.metrics.MeanAbsoluteError()(y_,class_preds_sum)

            val_accuracy += val_acc

    val_accuracy = round(val_accuracy / val_batch_size,5)
    val_accuracy_list.append(val_accuracy)
    print(val_accuracy)

    train_img_list, train_label_list = load_class_X_Y(total_path, 'train')

    if k == 50:
        best_accuracy = 0

    # tf.keras.models.save_model(model, "/media/crescom2/DATA/temp/")
    if val_accuracy >= best_accuracy:
        print('model save')
        best_accuracy = val_accuracy

        if k >= 50:
            model.save_weights('./last_epoch_' + str(k) + '_top_acc' + str(round(best_accuracy, 3)))
        else:
            model.save_weights('./epoch_' + str(k) + '_top_acc' + str(round(best_accuracy, 3)))

    plt.plot(train_accuracy_list, marker='', color='blue',
             label="train_accuracy" if k == 0 else "")
    plt.plot(train_loss_list, marker='', color='red',
             label="train_loss" if k == 0 else "")
    plt.plot(val_accuracy_list, marker='', color='green',
             label="val_accuracy" if k == 0 else "")
    plt.legend()
    plt.savefig('./' + 'train_plot.png')



