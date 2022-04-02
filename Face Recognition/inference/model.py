import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout, MaxPool2D
from tensorflow.keras import Model

class MyModel(Model):
  def __init__(self, number_of_classes):
    super().__init__()
    
    self.dense_1 = Dense(128, activation='relu')
    self.dense_2 = Dense(256, activation='relu')
    self.dense_3 = Dense(number_of_classes, activation='softmax')
    self.conv2d_1 = Conv2D(32, (3, 3), activation='relu', input_shape = (224, 224, 3))
    self.conv2d_2 = Conv2D(64, (5, 5), activation='relu')
    self.conv2d_3 = Conv2D(128, (5, 5), activation='relu')
    self.conv2d_4 = Conv2D(256, (3, 3), activation='relu')
    self.maxpool = MaxPool2D()
    self.flatten = Flatten()

  def call(self, x):
    x = self.conv2d_1(x)
    #x = self.conv2d_2(x)
    x = self.maxpool(x)
    x = self.conv2d_3(x)
    #x = self.conv2d_4(x)
    x = self.flatten(x)
    x = self.dropout1(x)
    #x = self.dense_1(x)
    #x = self.dense_2(x)
    out = self.dense_3(x)

    return out
