import os
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Dropout
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

# target_paths = ['./resrc/test/tmp/output_threshold/train', './resrc/test/tmp/output_threshold/test']

# for path in target_paths:
#     elem = os.listdir(path)
#     print(str(len(elem)) + ' files in ' + path + ' will be removed')
#     for e in elem:
#         os.remove(path + '/' + e)

base_path =  './resrc/test/sets/'
data_filter_type = ['no_filter', 'edges', 'threshold']#no_filter, edges, threshold
_path = base_path + data_filter_type[2]

train_path = _path + '/train'
test_path = _path + '/test'

print('train p : ' + train_path)
print('test p : ' + test_path)

train_data_size = len(listdir(train_path))
test_data_size = len(listdir(test_path))

print('train num : ', train_data_size)
print('test num : ', test_data_size)

# data set ------------------------------------------------------
np.random.seed(3)

train_datagen = ImageDataGenerator(1.0/255)

train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size = (64, 64),
        batch_size = 32,
        class_mode = 'binary')

test_datagen = ImageDataGenerator(1)

test_generator = test_datagen.flow_from_directory(
        test_path,
        target_size = (64, 64),
        batch_size = 32,
        class_mode = 'binary')

# model ------------------------------------------------------
model = Sequential()
model.add(Conv2D(1, kernel_size=(12, 12), # output = (8 * 8) * 3 * 32 = 6144 // kernel_size * channel * kernel_num
                 padding='same',
                 input_shape=(64, 64, 3),
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))# output = (8 * 8) * 3 * 64 = 6144
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
