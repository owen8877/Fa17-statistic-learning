import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import applications
import pandas as pd
from sklearn import preprocessing
from tqdm import *
import cv2

# dimensions of our images.
img_width, img_height = 224, 224

train_data_dir = '../data/dogs-breed/train'
test_data_dir = '../data/dogs-breed/test'
train_csv = '../data/dogs-breed/labels.csv'
test_csv = '../data/dogs-breed/sample_submission.csv'
batch_size = 8

df_train = pd.read_csv(train_csv)
df_test = pd.read_csv(test_csv)

train_number = df_train.shape[0]
test_number = df_test.shape[0]

# label process
label_onehot_encoder = preprocessing.LabelBinarizer()
train_labels_onehot = label_onehot_encoder.fit_transform(df_train['breed'])

def save_y(df, labels, save_filename):
    y_train = np.empty((df.shape[0], 120), dtype=np.uint8)
    i = 0
    for f, breed in tqdm(df.values):
        label = labels[i]
        y_train[i, ...] = label
        i += 1

    np.save(save_filename, y_train)


def read_x_img(data_dir, df, save_filename):
    x = np.empty((df.shape[0], 3, 224, 224), dtype=np.uint8)
    i = 0
    for f in tqdm(df['id']):
        img = cv2.resize(cv2.imread('{}/{}.jpg'.format(data_dir, f)).astype(np.uint8, copy=False),
                         (img_width, img_height)).transpose((2, 0, 1))
        x[i, ...] = img
        i += 1

    print(x.shape)
    x = x.transpose((0, 2, 3, 1))
    np.save(save_filename, x)
    return x


def load_predict(read_filename, save_filename, split_n):
    xs = np.array_split(np.load(read_filename), split_n)
    for i, x in enumerate(xs):
        x = np.array(x, dtype=np.float64, copy=False) - vgg_mean
        datagen = ImageDataGenerator(rescale=1. / 255)
        model = applications.VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
        generator = datagen.flow(x, y=None, batch_size=batch_size, shuffle=False)
        feature = model.predict_generator(generator)
        np.save(save_filename.format(i), feature)


def merge(read_filename, save_filename, split_n, total_sample_n):
    xs = np.empty((total_sample_n, 7, 7, 512), dtype=np.float64)
    total_progress = 0
    for i in range(split_n):
        x = np.load(read_filename.format(i))
        m = x.shape[0]
        xs[total_progress:total_progress+m] = x
        total_progress += m

    np.save(save_filename, xs)


vgg_mean = np.array([123.68, 116.779, 103.939]).reshape((1, 1, 1, 3))

# processing...
# save_y(df_train, train_labels_onehot, 'vgg16_y_train.npy')
# read_x_img(train_data_dir, df_train, 'vgg16_x_train_raw.npy')
# load_predict('vgg16_x_train_raw.npy', 'vgg16_x_train_feature_{}.npy', 8)
# merge('vgg16_x_train_feature_{}.npy', 'vgg16_x_train_feature.npy', 8, df_train.shape[0])

# read_x_img(test_data_dir, df_test, 'vgg16_x_test_raw.npy')
load_predict('vgg16_x_test_raw.npy', 'vgg16_x_test_feature_{}.npy', 10)
merge('vgg16_x_test_feature_{}.npy', 'vgg16_x_test_feature.npy', 10, df_test.shape[0])