import os, math, cv2, csv
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import imutils
from tensorflow.keras import layers
import tensorflow_addons as tfa

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

gpus = tf.config.experimental.list_physical_devices('GPU')
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


def load_cifar_all_X_Y():
    csv_path = '/media/crescom/새 볼륨1/dataset/trainLabels.csv'
    img_path = '/media/crescom/새 볼륨1/dataset/cifar10_train' + '/'
    img_list = []
    label_list = []
    one_hot_label_list = []

    with open(csv_path, 'r', encoding='utf-8') as csv_read:
        rdr = csv.reader(csv_read)
        for line in rdr:
            if 'label' not in line:
                img_list.append(img_path + line[0] + '.png')
                label_list.append(line[1])

        label_set = list(set(label_list))
        print(label_set)

        for one_label in label_list:
            one_hot_label_list.append(label_set.index(one_label))

    train_cnt = math.ceil(len(img_list) * 0.8)
    train_img_name, val_img_name = img_list[:train_cnt], img_list[train_cnt:]
    train_Y_list, val_Y_list = one_hot_label_list[:train_cnt], one_hot_label_list[train_cnt:]

    # print(img_list[0])
    # print(one_hot_label_list[0])

    return train_img_name, val_img_name, train_Y_list, val_Y_list


def load_class_X_Y(img_path, class_num):
    #   img_path = '/gdrive/My Drive/data/dog&cat/train/dog' + '/'
    total_img_name = os.listdir(img_path)[:1000]
    total_img_name = [os.path.join(img_path, x) for x in total_img_name]
    train_cnt = math.ceil(len(total_img_name) * 0.8)
    train_img_name, val_img_name = total_img_name[:train_cnt], total_img_name[train_cnt:]
    # print(len(total_img_name), len(train_img_name), len(val_img_name))

    total_val = []

    [total_val.append(class_num) for x in range(len(total_img_name))]
    train_Y_list, val_Y_list = total_val[:train_cnt], total_val[train_cnt:]

    return train_img_name, val_img_name, train_Y_list, val_Y_list


def one_hot_Y(train_Y_list, class_cnt):
    train_Y = np.eye(class_cnt)[train_Y_list]

    return train_Y


total_dog_path = '/media/crescom/새 볼륨1/dataset/dog_and_cat/dog' + '/'
total_cat_path = '/media/crescom/새 볼륨1/dataset/dog_and_cat/cat' + '/'

train_dog_img_name_def, val_dog_img_name_def, train_dog_Y, val_dog_Y = load_class_X_Y(total_dog_path, 0)
train_cat_img_name_def, val_cat_img_name_def, train_cat_Y, val_cat_Y = load_class_X_Y(total_cat_path, 1)

def random_shuffle(x, seed):
    random.seed(seed)
    random.shuffle(x)
    return x

total_train_X = train_dog_img_name_def + train_cat_img_name_def
total_train_X = random_shuffle(total_train_X, 2)
total_val_X = val_dog_img_name_def + val_cat_img_name_def
total_val_X = random_shuffle(total_val_X, 2)

total_train_Y = train_dog_Y + train_cat_Y
total_train_Y = random_shuffle(total_train_Y, 2)
total_val_Y = val_dog_Y + val_cat_Y
total_val_Y = random_shuffle(total_val_Y, 2)


# train_dog_Y, val_dog_Y = one_hot_Y(train_dog_Y, val_dog_Y, 2)
# train_cat_Y, val_cat_Y = one_hot_Y(train_cat_Y, val_cat_Y, 2)
#
# total_train_Y = np.concatenate([train_dog_Y, train_cat_Y],axis=0)
# total_val_Y = np.concatenate([val_dog_Y, val_cat_Y],axis=0)

# print('total_train_Y', total_train_Y)
# print('total_val_Y', total_val_Y)

# 슬라이스를 이용한 next batch
def next_batch(data_list, mini_batch_size, next_cnt):
    cnt = mini_batch_size * next_cnt
    batch_list = data_list[cnt:cnt + mini_batch_size]
    return batch_list


mini_batch_size = 15
batch_size = math.ceil(len(total_train_X) / mini_batch_size)
val_batch_size = math.ceil(len(total_val_X) / mini_batch_size)
print(batch_size)
batch_cnt = 0
val_batch_cnt = 0
epoch = 50
max_val_acc = 0

# 마지막 슬라이스에 오면 처음부터 img read
# for i in range(10):
#     print(next_batch(total_train_X[:8], mini_batch_size, batch_cnt))
#
#     print(mini_batch_size * (batch_cnt + 1))
#     if mini_batch_size * (batch_cnt + 1) >= len(total_train_X[:8]):
#         batch_cnt = 0
#     else:
#         batch_cnt += 1

# batch마다 img를 list에 담아 return
def img_loader(img_list, rot_rate=0.0, shift_rate=0.0, aug=True):
    img_set = []
    for one_img_list in img_list:
        one_img = cv2.imread(one_img_list)

        if aug:
            filp_ran = random.randint(0, 1)
            rot = random.randint(0, rot_rate)

            x, y = one_img.shape[1], one_img.shape[0]

            shift_x = random.randint(-int(x * shift_rate), int(x * shift_rate))
            shift_y = random.randint(-int(y * shift_rate), int(y * shift_rate))

            if filp_ran:
                one_img = cv2.flip(one_img, 1)

            one_img = imutils.translate(one_img, shift_x, shift_y)
            one_img = imutils.rotate(one_img, rot)

        one_img = cv2.resize(one_img, (224, 224))

        #     one_img = cv2.imread(one_img_list,0)
        #     print(one_img.shape)
        img_set.append(one_img)
    return np.array(img_set, dtype=np.float32)


print('total_train_Y', len(total_train_Y))

# print('len(total_train_X[:8]', len(total_train_X[:8]))

# for i in range(3):
#   one_batch_X_list = next_batch(total_train_X[:8],mini_batch_size,batch_cnt)
#   one_batch_Y = next_batch(total_train_Y[:8],mini_batch_size,batch_cnt)
#   one_batch_X = img_loader(one_batch_X_list)

#   print('one_batch_X', one_batch_X.shape)
#   print('one_batch_Y', one_batch_Y.shape)

# #   for k in one_batch_img:
# #     plt.imshow(k)
# #     plt.show()

#   print('batch_size * (batch_cnt+1)',batch_size * (batch_cnt+1))
#   if mini_batch_size * (batch_cnt + 1) >= len(total_train_X[:8]):
#     batch_cnt = 0
#   else:
#     batch_cnt += 1

l2_regul = tf.keras.regularizers.l2(l=0.1)
he_init = tf.keras.initializers.he_normal()

# def conv2d(filters,kernel_size):
#     layer = layers.Conv2D(filters=filters,
#                   kernel_size=(kernel_size,kernel_size),
#                   padding='same',
#                   activation='relu',
#                   # kernel_regularizer=l2_regul,
#                   kernel_initializer=he_init,
#                   strides=1)
#     return layer

# custom model
# model = tf.keras.Sequential([layers.Conv2D(filters=64,
#                                         kernel_size=(3,3),
#                                         padding='same',
#                                         input_shape=(224, 224, 3),
#                                         activation='relu',
#                                         kernel_regularizer=l2_regul,
#                                         kernel_initializer=he_init,
#                                         strides=1,
#                                         ),
#                              conv2d(128,3),
#                              tf.keras.layers.MaxPooling2D(padding='same'),
#                              conv2d(256, 3),
#                              tf.keras.layers.MaxPooling2D(padding='same'),
#                              conv2d(512, 3),
#                              tf.keras.layers.MaxPooling2D(padding='same'),
#                              conv2d(1024, 3),
#                              tf.keras.layers.GlobalAveragePooling2D(),
#                              tf.keras.layers.Dense(2,activation='softmax',kernel_regularizer=l2_regul,kernel_initializer=he_init)
# ])

# input = tf.keras.Input(shape=[24, 24, 3])  # change this shape to [None,None,3] to enable arbitraty shape input
# a = tf.keras.layers.Conv2D(28, (3, 3), strides=1, padding='valid', name='conv1')(input)
# x = tf.keras.layers.PReLU(shared_axes=[1, 2], name='prelu1')(a)
# x = tf.keras.layers.MaxPool2D(pool_size=3,strides=2, padding='same')(x)
#
# x = tf.keras.layers.Conv2D(48, (3, 3), strides=1, padding='valid', name='conv2')(x)
# x = tf.keras.layers.PReLU(shared_axes=[1, 2], name='prelu2')(x)
# x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2)(x)
#
# x = tf.keras.layers.Conv2D(64, (2, 2), strides=1, padding='valid', name='conv3')(x)
# x = tf.keras.layers.PReLU(shared_axes=[1, 2], name='prelu3')(x)
# x = tf.keras.layers.GlobalAveragePooling2D()(x)
# x = tf.keras.layers.Dense(128, name='conv4')(x)
# x = tf.keras.layers.Softmax(name='prelu4')(x)
# classifier = tf.keras.layers.Dense(2, activation='softmax', name='conv5-1')(x)
#
# model = tf.keras.Model([input], [classifier])

# pretrained model

# tf.keras.losses.Huber

IMG_SHAPE = (224,224,3)
base_model = tf.keras.applications.ResNet101(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')

base_model.trainable = True
x = base_model.output

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = tf.keras.layers.Dense(2, activation='softmax',kernel_initializer=he_init)

model = tf.keras.Sequential([
  base_model,
  global_average_layer,
  prediction_layer
])

# tf.keras.models.load_model('/media/crescom2/DATA/temp')
print('load model')

model.summary()

optimizer = tf.optimizers.Adam(learning_rate=0.0001)

for k in range(epoch):
    print('epoch : ', k)
    val_accuracy = 0
    train_accuracy = 0
    train_loss = 0

    epoch_loss = tf.keras.metrics.Mean()
    epoch_acc = tf.keras.metrics.CategoricalAccuracy()

    val_epoch_loss = tf.keras.metrics.Mean()
    val_epoch_acc = tf.keras.metrics.CategoricalAccuracy()

    for i in range(batch_size):
        train_one_batch_X_list = next_batch(total_train_X, mini_batch_size, i)
        # print('train one_batch_X_list', one_batch_X_list)
        train_one_batch_Y = next_batch(total_train_Y, mini_batch_size, i)
        #             print('one_batch_Y',one_batch_Y)
        train_one_batch_Y = one_hot_Y(train_one_batch_Y, 2)
        train_one_batch_X = img_loader(train_one_batch_X_list, 45, 0.1, True)
        # print(one_batch_X.shape)
        # print(one_batch_X_list)
        # print(one_batch_Y)

        #             print('train one_batch_X', one_batch_X.shape)
        #             print('train one_batch_Y', one_batch_Y.shape)

        with tf.GradientTape() as tape:
            y_ = model(train_one_batch_X)
            # loss = SparseCategoricalCrossentropy(from_logits= True)
            
            loss = tf.losses.categorical_crossentropy(train_one_batch_Y, y_)

            grad = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grad, model.trainable_variables))

        acc = np.mean(np.cast[np.int32](np.equal(np.argmax(y_,axis=1), np.argmax(train_one_batch_Y,axis=1))))
        train_loss += np.mean(loss)
        train_accuracy += acc

        epoch_loss(loss)  # 현재 배치 손실을 추가합니다.
        # 예측된 레이블과 실제 레이블 비교합니다.
        epoch_acc(train_one_batch_Y, y_)

        # train_loss_results.append(epoch_loss.result())
        # train_accuracy_results.append(epoch_acc.result())

        # print("에포크 {:03d}: 손실: {:.3f}, 정확도: {:.3%}".format(k,
        #                                                    np.mean(loss),
        #                                                    acc))

        # print('epoch loss : ',train_loss / batch_size, 'epoch acc : ', train_accuracy / batch_size)

    ranint = random.randint(1,100)
    total_train_X = random_shuffle(total_train_X, ranint)
    total_train_Y = random_shuffle(total_train_Y, ranint)

    print(train_accuracy / batch_size)
    print(train_loss / batch_size)

    # print('batch : ', batch_cnt, 'train_acc : ', one_batch_train_acc, 'train loss : ', loss)

    for i in range(val_batch_size):
        val_one_batch_X_list = next_batch(total_val_X, mini_batch_size, i)
        # print('val one_batch_X_list', val_one_batch_X_list)
        val_one_batch_Y = next_batch(total_val_Y, mini_batch_size, i)
        val_one_batch_Y = one_hot_Y(val_one_batch_Y, 2)
        val_one_batch_X = img_loader(val_one_batch_X_list, 0, 0, aug=False)

        y_ = model(val_one_batch_X)
        # loss = tf.keras.losses.categorical_crossentropy(y_true=one_batch_Y, y_pred=y_)
        loss = tf.losses.categorical_crossentropy(val_one_batch_Y,y_)

        #             print('loss', loss)
        #             print('probability', prob)
        #             print('label', label)

        # print(y_)
        # np.mean(loss))

        val_acc = np.mean(np.cast[np.int32](np.equal(np.argmax(y_, axis=1), np.argmax(val_one_batch_Y, axis=1))))
        # print(val_acc)

        val_accuracy += val_acc

    print(val_accuracy / val_batch_size)

    # tf.keras.models.save_model(model, "/media/crescom2/DATA/temp/")

