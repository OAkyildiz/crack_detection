#!/usr/bin/env python
# Prepaere the dataset for CNNoperations with Keras.
# This is intentionally seperate from the rest of the dataset_tools
# as it pertains to modifying an existing dataset to use with Keras API
from keras.preprocessing.image import ImageDataGenerator
 #
 # Just a layout as a mental note:
 # Also maybe rename  after labeling?
 # data/
 #     train/
 #         positive/
 #             crack_1.jpg
 #             crack_3.jpg
 #             ...
 #         negative/
 #             crack_1.jpg
 #             crack_3.jpg
 #             ...
 #     test/
 #         positive/
 #             crack_4jpg
 #             crack_2.jpg
 #             ...
 #         negative/
 #             crack_4jpg
 #             crack_2.jpg
 #             ...

def gen_set(folder, )
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 2000
nb_validation_samples = 800
epochs = 50
batch_size = 16

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
input_shape = (img_width, img_height, 3)

test_datagen = ImageDataGenerator(rescale=1. / 255)
###
#
# MODEL was here
#
###
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)


    
