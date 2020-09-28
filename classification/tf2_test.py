import math, csv
from keras import applications
import tensorflow as tf
import random
import imutils, os, cv2
import numpy as np
from image_util import *
from sklearn.metrics import confusion_matrix

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
input_size = 331
ckpt_path = '/media/crescom/새 볼륨1/ckpt/AS/2020_09_25_copy' + '/'
model_name = 'top_acc0.77'
mini_batch_size = 1

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

def load_class_X_Y(img_path, set_name):
    top_img_path = img_path + '/' + set_name + '/'
    class_name = os.listdir(top_img_path)

    img_list = []
    label_list = []

    for one_class in range(len(class_name)):
        one_class_path = top_img_path + class_name[one_class] + '/'
        one_class_img_name = os.listdir(one_class_path)
        if class_name[one_class] == '0':
            shuffle_int = random.randint(0,100)
            random.seed(shuffle_int)
            random.shuffle(one_class_img_name)
            for one_class_one_img_name in one_class_img_name[:4000]:
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

val_img_list, val_label_list = load_class_X_Y('/media/crescom/새 볼륨1/dataset/spine/crop_img/0.1_final_up_down/train_set/down', 'val')

filenames = []
for i in val_img_list:
    frist_idx = i.rindex('/')
    frist = i[frist_idx:]
    second_idx = i[:frist_idx].rindex('/')
    second = i[:frist_idx][second_idx+1:]
    filename = second + frist
    filenames.append(filename)

# train_img_list, train_label_list = train_img_list[:100], train_label_list[:100]
# val_img_list, val_label_list = val_img_list[:100], val_label_list[:100]

# print(val_img_list[:-10], val_label_list[:-10])

# 슬라이스를 이용한 next batch
def next_batch(data_list, mini_batch_size, next_cnt):
    cnt = mini_batch_size * next_cnt
    batch_list = data_list[cnt:cnt + mini_batch_size]
    return batch_list

val_batch_size = math.ceil(len(val_img_list) / mini_batch_size)
batch_cnt = 0
val_batch_cnt = 0
max_val_acc = 0

def img_loader(img_list, rot_rate=0, shift_rate=0.0, aug=''):
    img_set = []
    for one_img_list in img_list:
        one_img = cv2.imread(one_img_list)

        if aug == 'train':
            img_preprocessing = random.randint(0, 3)
            one_img = clahe(one_img) if img_preprocessing == 0 else (normalize(one_img) if img_preprocessing == 1 else
                                                                     (sharpen(
                                                                         one_img) if img_preprocessing == 2 else one_img))

            filp_ran = random.randint(0, 1)
            rot = random.randint(0, rot_rate)

            x, y = one_img.shape[1], one_img.shape[0]

            shift_x = random.randint(-int(x * shift_rate), int(x * shift_rate))
            shift_y = random.randint(-int(y * shift_rate), int(y * shift_rate))

            if filp_ran:
                # vertical flip
                one_img = cv2.flip(one_img, 1)

            one_img = imutils.translate(one_img, shift_x, shift_y)
            one_img = imutils.rotate(one_img, rot)
        elif aug == 'ensemble_test':
            clahe_img = cv2.resize(clahe(one_img), (input_size, input_size))
            normal_img = cv2.resize(normalize(one_img), (input_size, input_size))
            sharpen_img = cv2.resize(sharpen(one_img), (input_size, input_size))
            one_img = cv2.resize(one_img, (input_size, input_size))

            img_set = np.stack([one_img, clahe_img, normal_img, sharpen_img], axis=0)
            img_set = img_set.astype(np.float32)
            img_set /= 127.5
            img_set -= 1.
            return img_set

        one_img = cv2.resize(one_img, (input_size, input_size))

        img_set.append(one_img)

    img_set = np.array(img_set, dtype=np.float32)

    img_set /= 127.5
    img_set -= 1.

    return img_set

l2_regul = tf.keras.regularizers.l2(l=0.1)
he_init = tf.keras.initializers.he_normal()

IMG_SHAPE = (input_size,input_size,3)

base_model = tf.keras.applications.NASNetLarge(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet',)

rescale = tf.keras.applications.nasnet.preprocess_input
base_model.trainable = False
model_output = base_model.output

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = tf.keras.layers.Dense(4, activation='softmax',kernel_initializer=he_init)

model = tf.keras.Sequential([
  base_model,
  global_average_layer,
  prediction_layer
])

model.summary()

# tf.keras.models.load_model('/media/crescom2/DATA/temp')

print('load model')
model.load_weights(ckpt_path + model_name)

val_accuracy = 0
predict_list = []

for i in range(val_batch_size):
    print(i)
    val_one_batch_X_list = next_batch(val_img_list, mini_batch_size, i)
    # print('val one_batch_X_list', val_one_batch_X_list)
    val_one_batch_Y = next_batch(val_label_list, mini_batch_size, i)
    val_one_batch_Y = one_hot_Y(val_one_batch_Y, 4)
    val_one_batch_X = img_loader(val_one_batch_X_list, 0, 0, aug='')


    y_ = model(val_one_batch_X)

    class_idx = np.argmax(y_, axis=1)
    class_preds_sum = np.sum(y_, axis=0) / 4

    ensemble_predict = np.argmax(class_preds_sum)
    print('ensemble_predict',ensemble_predict)
    y = np.argmax(val_one_batch_Y)

    val_acc = np.mean(np.cast[np.int32](np.equal(y, ensemble_predict)))

    predict_list.append(ensemble_predict)


    # batch_predict = np.argmax(y_, axis=1)
    # batch_true = np.argmax(val_one_batch_Y, axis=1)
    #
    # val_acc = np.mean(np.cast[np.int32](np.equal(batch_predict, batch_true)))
    # for i in range(len(batch_predict)):
    #     predict_list.append(batch_predict[i])

    val_accuracy += val_acc

print('accuracy :', val_accuracy / val_batch_size)
confusion = confusion_matrix(val_label_list, predict_list)

incorrect = []
incorrect_list = []

for i in range(len(predict_list)):
   if not predict_list[i] == val_label_list[i]:
       incorrect.append(filenames[i])

print(incorrect)

# for i in incorrect:
#     incorrect_list.append(image_files[i])
print('gs / predict')
print(confusion)

f = open(ckpt_path + '/' + 'val_result.txt', 'w')
for i in range(len(filenames)):
   f.write('filename : ' + filenames[i] + '\n')
   f.write('pred : ' + str(predict_list[i]) + '\n')
   f.write('true : ' + str(val_label_list[i]) + '\n')

f.write('\nincorrect_list' + '\n')
f.write(str(incorrect) + '\n\n')
f.write('confusion' + '\n')
f.write('gs / predict' + '\n')
f.write(str(confusion) + '\n\n')
f.write('total_accuracy' + '\n')
f.write(str(val_accuracy / val_batch_size) + '\n\n')

f.close()


