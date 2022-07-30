import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
from keras.datasets import fashion_mnist

seed = 7
numpy.random.seed(seed)
#加载数据
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')

X_train = X_train / 255
X_test = X_test / 255

# 对输出进行one hot编码
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

# MLP模型
def baseline_model():
model = Sequential()
model.add(Dense(num_pixels, input_dim=num_pixels, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
return model

# 建立模型
model = baseline_model()

# Fit
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)

#Evaluation
scores = model.evaluate(X_test, y_test, verbose=0)
print("使用MLP并迭代十次的正确率为:" ,'%.2f' %(scores[1]*100) , "%")

