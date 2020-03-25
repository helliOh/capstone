#교수님 예제
from os import listdir
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dropout
import numpy as np
import tensorflow as tf

base_path =  './resrc/test/sets/'   # 여기 수정하고 <-----------------------------
data_filter_type = ['no_filter', 'edges', 'threshold']#no_filter, edges, threshold
base_path = base_path + data_filter_type[1]
train_path = base_path + '/train'
test_path = base_path + '/test'
print('train p : ' + train_path)
print('test p : ' + test_path)

train_data_size = len(listdir(train_path))
test_data_size = len(listdir(test_path))

print('train num : ', train_data_size)
print('test num : ', test_data_size)

# data set ------------------------------------------------------
np.random.seed(3)

x_size = 64
y_size = 64
x_pool_size = 2
y_pool_size = 2

train_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size = (x_size, y_size),
        batch_size = 32,
        class_mode = 'binary')
                 
test_datagen = ImageDataGenerator(1)

test_generator = test_datagen.flow_from_directory(
        test_path,
        target_size = (x_size, y_size),
        batch_size = 32,
        class_mode = 'binary')


    # model ------------------------------------------------------
model = Sequential()
model.add(Conv2D(1, kernel_size=(12, 12),
                 padding='same',
                 input_shape=(x_size,y_size,3),
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(x_pool_size, y_pool_size))) #output size = 120 * 120
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# model  compile ------------------------------------------------------
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# model training ------------------------------------------------------
model.fit_generator(
        train_generator,
        epochs=20,
        validation_data=test_generator)

# model evaluating ------------------------------------------------------
print("-- Evaluate --")

scores = model.evaluate_generator(test_generator)

print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))
