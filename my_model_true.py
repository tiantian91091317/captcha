import numpy as np
import glob
from PIL import Image
from preprocess import Vocab
import tensorflow as tf
from keras.callbacks import TensorBoard  
from keras.callbacks import ModelCheckpoint

samples = glob.glob('data/train/*.jpg')
np.random.shuffle(samples)
# print(samples)

def CaptchaGenerator(samples, batch_size):
    # to determine dimensions
    # 
    while True:
        batch = np.random.choice(samples, batch_size)
        X = []
        y = []
        for sample in batch:
            img = np.asarray(Image.open(sample))
            text = Vocab().text_to_one_hot(sample[-8:-4])
            X.append(img)
            y.append(text)

        X = np.asarray(X)
        y = np.asarray(y)            
        # print("data:")
        # print(X.shape)
        # print(y.shape)
        
        yield X, y


def CaptchaGenerator4(samples, batch_size):
    # to determine dimensions
    # 
    while True:
        batch = np.random.choice(samples, batch_size)
        X = []
        y = []
        for sample in batch:
            img = np.asarray(Image.open(sample))
            text = Vocab().text_to_one_hots(sample[-8:-4])
            X.append(img)
            y.append(text)
        X = np.asarray(X)
        y = np.asarray(y)
        for i in range(4):
            print(y[:,i]) 
        yield X, [y[:,i] for i in range(4)]


from keras.layers import Dense, Dropout, Activation, Flatten, Input, merge, AveragePooling2D
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Model, Sequential
import warnings
warnings.filterwarnings('ignore')

batch_size = 2
img_shape = (60, 240, 3) # height, width, channels
nb_classes = 62

# 用创哥的网络结构，第二版 sigmoid 方案
# 卷积（relu）、池化（dropout）
inputs = Input(shape = img_shape)
conv1 = Conv2D(32, (3, 3))(inputs)
relu1 = Activation('relu')(conv1)
pool1 = MaxPooling2D(pool_size=(2,2), border_mode='same')(relu1)
drop1 = Dropout(0.2)(pool1)
# 卷积（relu）、池化（dropout）
conv2 = Conv2D(32, (3, 3))(drop1)
relu2 = Activation('relu')(conv2)
pool2 = MaxPooling2D(pool_size=(2,2), border_mode='same')(relu2)
drop2 = Dropout(0.2)(pool2)

# 拉平，全连接
fl = Flatten()(drop2)
fc1 = Dense(1024, name = 'fc1')(fl)
drop3 = Dropout(0.2)(fc1)


# 一直跑不成功的 第一版 sigmoid 方案
# inputs = Input(shape = img_shape, name = "inputs")
# conv1 = Conv2D(32, (3, 3), name = "conv1")(inputs)
# relu1 = Activation('relu', name="relu1")(conv1)
# drop1 = Dropout(0.2, name = 'dropout1')(relu1)
# conv2 = Conv2D(32, (3, 3), name = "conv2")(drop1)
# relu2 = Activation('relu', name="relu2")(conv2)
# drop2 = Dropout(0.2, name = 'dropout2')(relu2)
# pool2 = MaxPooling2D(pool_size=(2,2), border_mode='same', name="pool2")(drop2)
# conv3 = Conv2D(64, (3, 3), name = "conv3")(pool2)
# relu3 = Activation('relu', name="relu3")(conv3)
# pool3 = AveragePooling2D(pool_size=(2,2), name="pool3")(relu3)
# fl = Flatten()(pool3)

# 最早的 用 concatenate/merge 的方案
# fc1 = Dense(nb_classes, name="fc1")(fl)
# drop = Dropout(0.25, name = "dropout1")(fc1)
# fc = Dense(nb_classes*4, name = 'fc', activation = 'sigmoid')(fl)
# fc21 = Dense(nb_classes, name="fc21", activation="softmax")(drop)
# fc22 = Dense(nb_classes, name="fc22", activation="softmax")(drop)
# fc23 = Dense(nb_classes, name="fc23", activation="softmax")(drop)
# fc24 = Dense(nb_classes, name="fc24", activation="softmax")(drop)

# 输出sigmoid
fc2 = Dense(nb_classes*4, name = 'fc', activation = 'sigmoid')(drop3)

def custom_accuracy(y_true, y_pred):
    predict = tf.reshape(y_pred, [-1, 4,62])
    max_idx_p = tf.argmax(predict, 2)#这个做法牛逼，不用再做stack和reshape了，2，是在Charset那个维度上
    max_idx_l = tf.argmax(tf.reshape(y_true, [-1, 4,62]), 2)     
    correct_pred = tf.equal(max_idx_p, max_idx_l)
    _result = tf.map_fn(fn=lambda e: tf.reduce_all(e),elems=correct_pred,dtype=tf.bool)
    return tf.reduce_mean(tf.cast(_result, tf.float32))

checkpoint = ModelCheckpoint('best.hdf5', save_best_only=True, 
        monitor='val_loss', mode='min')

model = Model(inputs = inputs, outputs = fc2)
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy',custom_accuracy]
              )
model.summary()
model.fit_generator(CaptchaGenerator(samples, batch_size), 
    steps_per_epoch=len(samples)/batch_size, 
    callbacks=[TensorBoard(log_dir='./log'),checkpoint],
    epochs=10)



# 创哥帮忙改的 用softmax的方案
# from keras.layers import Input, Dense, Layer
# target_input = Input(shape=(62*4,))
# class SampledSoftmax(Layer):
#     def __init__(self, **kwargs):
#         super(SampledSoftmax, self).__init__(**kwargs)


#     def call(self, inputs):
#         fc1,fc2,fc3,fc4 = tf.split(inputs,[62,62,62,62],1)
#         c1 = tf.keras.backend.softmax(fc1)
#         c2 = tf.keras.backend.softmax(fc2)
#         c3 = tf.keras.backend.softmax(fc3)
#         c4 = tf.keras.backend.softmax(fc4)
#         return [c1,c2,c3,c4]

# fc = Dense(nb_classes*4, name = 'fc', activation = 'relu')(fl)
# outputs = fc()
# cs = SampledSoftmax()([fc,target_input])

# from keras.objectives import categorical_crossentropy
# def custom_loss(y_true, y_pred):
#     return tf.reduce_mean(categorical_crossentropy(y_true, y_pred))

# model = Model(inputs = inputs, outputs = cs)
# model.compile(loss=custom_loss,
#               optimizer='adam',
#               metrics=['accuracy'])
# model.summary()
# model.fit_generator(CaptchaGenerator4(samples, batch_size), steps_per_epoch=len(samples)/batch_size, epochs=1)




model.save('capcha_model.h5')


