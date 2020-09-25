import nibabel as nib
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# path = '/media/새 볼륨1/dataset/COVID_data/COVID_CT/COVID19_1110/studies/CT-0/study_0001.nii.gz'
# path = '/media/새 볼륨1/dataset/COVID_data/COVID_CT/zenodo_set/lung_and_infection_mask/coronacases_001.nii.gz'
path = '/media/새 볼륨1/dataset/COVID_data/COVID_CT/medical_segmentation/rp_msk/1.nii.gz'

img = nib.load(path).get_data()
print(img.shape)

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

img = np.expand_dims(img,axis=0)
img = np.expand_dims(img,axis=4)
img = np.cast[np.float32](img)
print(img.shape)
# layer = tf.keras.Input(shape=[1,512,512,43])(img)
# print(layer.shape)
layer = tf.keras.layers.Conv3D(3,2,padding='same')(img)
print(layer.shape)
layer = tf.keras.layers.Conv3D(3,2,strides=(2,2,2),padding='same')(layer)
print(layer.shape)

#
# for i in range(len(img[2])):
#     plt.imshow(img[:,:,i])
#     plt.show()
