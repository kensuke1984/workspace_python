import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.layers import Conv2D,MaxPooling2D
from keras import backend as K
import matplotlib.pyplot as plt


batch_size = 128
num_classes = 10
epochs = 12

img_rows, img_cols = 28,28

# (x_train, y_train),(x_test,y_test) mnist.load_data()

# if K.image_data_format()=='channels_first':
