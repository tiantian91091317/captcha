from keras.models import Model, load_model
import glob
import numpy as np
from PIL import Image
from preprocess import Vocab
import matplotlib.pyplot as plt # plt 用于显示图片


model = load_model('capcha_model_one_char.h5')
samples = glob.glob('data/test/*.jpg')

batch_size = 10
batch = np.random.choice(samples, batch_size)
print(batch)

for sample in batch:
    img = np.asarray(Image.open(sample)).reshape(1,60,240,3)
    plt.imshow(Image.open(sample))

    text = sample[-8:-4]
    pre_text = Vocab().one_hot_to_text(model.predict(img)[0])
    print(model.predict(img))
    print('prediction is:{}'.format(pre_text))
    print('real is {}'.format(text))


# batch = np.random.choice(samples, batch_size)
# X = []
# y = []
# for sample in batch:
#     img = np.asarray(Image.open(sample))
#     text = Vocab().text_to_one_hot(sample[-8:-7])
#     X.append(img)
#     y.append(text)
# X = np.asarray(X)
# y = np.asarray(y)
# # y1 = [y[:,i] for i in range(4)]
# # print(y1)

# for yy in y1:
#     print(yy.shape)

# X = []
# y = []
# for sample in batch:
#     img = np.asarray(Image.open(sample))
#     text = Vocab().text_to_one_hot(sample[-8:-4])
#     print(text)
#     X.append(img)
#     y.append(text)

# X = np.asarray(X)
# y = np.asarray(y)
# print('\n')
# print(X)
# print(y)

# text = Vocab().text_to_one_hot(sample[-8:-4])


#     X.append(img)
#     y.append(text)
# X = np.asarray(X)
# y = np.asarray(y)



# model = load_model('capcha_model.h5')
# import math
# plt.figure(figsize=(15,5))
# acc = 0;
# n = 20
# num_per_row = 5
# num_rows = math.ceil(n/float(num_per_row))

# for i in range(n):
#     img, text = JrttCaptcha().get_captcha()
#     X = np.empty((1, 30, 120, 3))
#     X[0] = np.array(img, dtype = np.uint8) / 255
#     Y_pred = model.predict(X)
#     Pred_text = Vocab().one_hot_to_text(Y_pred[0])
#     if Pred_text == text:
#         acc +=1

#     if Pred_text != text:
#         print("True value is ", text)
#         print("Prediceted value is ", Pred_text)

#     ax = plt.subplot(num_rows, num_per_row, i+1)
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
#     plt.imshow(img)
#     plt_title = plt.title("%s" % Pred_text)
#     if Pred_text != text:
#         plt.setp(plt_title, color='r')

# print("Accuracy is: %.4f" % (acc/float(n)))