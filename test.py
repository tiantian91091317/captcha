from keras.layers import Dense, Dropout, Activation, Flatten, Input, _Merge, AveragePooling2D
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Model, Sequential
import warnings
warnings.filterwarnings('ignore')

import numpy as np
# from generator import JrttCaptchaGenerator
# from captcha import JrttCaptcha
# from vocab import Vocab

batch_size = 32
ocr_shape = (30, 120, 3) # height, width, channels
nb_classes = 62

inputs = Input(shape = ocr_shape, name = "inputs")
conv1 = Conv2D(32, (3, 3), name = "conv1")(inputs)
relu1 = Activation('relu', name="relu1")(conv1)
conv2 = Conv2D(32, (3, 3), name = "conv2")(relu1)
relu2 = Activation('relu', name="relu2")(conv2)
pool2 = MaxPooling2D(pool_size=(2,2), border_mode='same', name="pool2")(relu2)
conv3 = Conv2D(64, (3, 3), name = "conv3")(pool2)
relu3 = Activation('relu', name="relu3")(conv3)
pool3 = AveragePooling2D(pool_size=(2,2), name="pool3")(relu3)
fl = Flatten()(pool3)
fc1 = Dense(nb_classes, name="fc1")(fl)
drop = Dropout(0.25, name = "dropout1")(fc1)
fc21 = Dense(nb_classes, name="fc21", activation="softmax")(drop)
fc22 = Dense(nb_classes, name="fc22", activation="softmax")(drop)
fc23 = Dense(nb_classes, name="fc23", activation="softmax")(drop)
fc24 = Dense(nb_classes, name="fc24", activation="softmax")(drop)

merged = _Merge([fc21, fc22, fc23, fc24], mode = 'concat', name = "merged")
print('success')


