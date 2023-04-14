import os
import cv2
import numpy as np
import random

import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split


"""
    This script is to create a simple inpainting model, meaning a model which can take a masked image (so an 
    image with several parts missing) and outputs the same image but with the missing parts filled in. In order to
    run it, you have to have access to a large dataset and store it somewhere on your computer (and refer to it
    in pathToImgFold). For example, you could use the Google open images dataset (more information at 
    https://storage.googleapis.com/openimages/web/index.html). 
    
    If you have limited memory or no access to a large dataset of images, you could use the cifar dataset, which
    consists of random images of size (32x32x3). You can use the following code to download that.
    
    (x_train, _), (x_test, _) = keras.datasets.cifar10.load_data()

    Note that our Data generator accepts a list of paths to images, so we don't have to load all of them at the same
    time. So if you want to use the cifar10 set, you either have to save the images in a sepperate folder or alter
    the data generator to accept images instead of paths to images. 

    author : Martijn Folmer
    Date : 14-04-2023

"""

# Data from user
DimSize = 224                                   # The model will resize the image to a square of size (DimSize, DimSize, 3)
MaxImages = 50000                               # Set the maximum images
BatchSize = 32                                  # batch size of each training step
N_epochs = 20                                   # how many epochs we want to train
pathToImgFold = 'PATH/TO/TRAINING/IMAGES'       # Path to the folder containing all of the subfolders which contain the training images
pathToResultsFold = 'PATH/TO/RESULTING/IMAGES'  # Path to the folder where we want to store the testing images we generate at end
pathToSavedModelFold = 'PATH/TO/SAVED/MODEL'    # Path to folder where we save the model

# Loading all of the images (assumes that the images are in subfolders inside of pathToImgFold
all_fold = [pathToImgFold + f"/{fold}" for fold in os.listdir(pathToImgFold)]
all_imgPath = []
for fold in all_fold:
    all_imgPath.extend([fold + f"/{f}" for f in os.listdir(fold)])
print(f"Total number of potential images : {len(all_imgPath)}")

random.shuffle(all_imgPath)             # shuffle it
all_imgPath = all_imgPath[:MaxImages]   # Make sure that we have a maximum number of images

# Print split list of Image paths to X_train and X_test, so we have 0.8 training data and 0.2 Testing data
x_train, x_test = train_test_split(all_imgPath, test_size=0.2)
x_train = np.asarray(x_train)
x_test = np.asarray(x_test)
print(f"Number of training image Paths : {x_train.shape[0]}")
print(f"Number of testing image Paths : {x_test.shape[0]}")


# Dataset augmentation class
class DataAugmentation(keras.utils.Sequence):
    """
        This creates generators which returns combination Maksed images and accompanied filled in images
    """

    def __init__(self, ListOfImagePaths, batch_size=32, dim=(64, 64), n_channels=3, shuffle=True):
        self.batch_size = batch_size    # The number of images to return for each sample
        self.X = ListOfImagePaths       # input and output both use the same list of images
        self.y = ListOfImagePaths
        self.dim = dim                  # Dimension of image (what we will resize to it)
        self.n_channels = n_channels    # channel dimension (3 for RGB, 1 for grayscale)
        self.shuffle = shuffle          # whether we shuffle the indexes or not
        self.on_epoch_end()             # initialize the shuffled indexes

    def __len__(self):
        return int(np.floor(len(self.X) / self.batch_size))

    def __getitem__(self, index):
        # Get a list of indexes which represent which images we return this batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        X_batch, y_batch = self.data_generation(indexes)

        return X_batch, y_batch

    def on_epoch_end(self):
        # Function fires every time that an epoch ends.
        self.indexes = np.arange(len(self.X))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def data_generation(self, idxs):

        X_batch = np.empty((self.batch_size, self.dim[0], self.dim[1], self.n_channels))  # Masked images
        y_batch = np.empty((self.batch_size, self.dim[0], self.dim[1], self.n_channels))  # Original images

        # Iterate through random indexes
        for i, idx in enumerate(idxs):
            image_copy = cv2.imread(self.X[idx])
            image_copy = cv2.resize(image_copy, self.dim)
            y_copy = image_copy.copy()

            masked_image = self.CreateMask(image_copy)

            y_batch[i] = y_copy / 255
            X_batch[i] = masked_image / 255

        return X_batch, y_batch

    def CreateMask(self, img):

        # This will create the masked image, either with random lines, or rectangles
        # Mask has the same dimensions as the image
        mask = np.full((self.dim[0], self.dim[1], self.n_channels), 255, np.uint8)

        # Random lines
        if random.random() < 0.5:
            for _ in range(np.random.randint(1, 10)):  # between 1 and 10 lines
                x1, x2 = np.random.randint(1, self.dim[0]), np.random.randint(1, self.dim[0])
                y1, y2 = np.random.randint(1, self.dim[1]), np.random.randint(1, self.dim[1])
                thickness = np.random.randint(5, 15)   # random thickness of the lines
                cv2.line(mask, (x1, y1), (x2, y2), (1, 1, 1), thickness)
        # Rectangles
        else:
            for _ in range(np.random.randint(1, 3)):   # between 1 and 3 rectangles
                x1, y1 = np.random.randint(1, self.dim[0]-1), np.random.randint(1, self.dim[1]-1)
                x2, y2 = x1 + np.random.randint(50, 75), y1 + np.random.randint(50, 75)
                x2, y2 = min(x2, self.dim[0]-1), min(y2, self.dim[1]-1)

                cv2.rectangle(mask, (x1, y1), (x2, y2), (1, 1, 1), -1)

        # Bitwise operation to create the masked image
        masked_image = cv2.bitwise_and(img, mask)

        return masked_image

# Create our training generator and test generator, which return Masked image - image combinations
traingen = DataAugmentation(x_train, dim=(DimSize, DimSize))
testgen = DataAugmentation(x_test, dim=(DimSize, DimSize), shuffle=False)

# Metric for comparing images
def dice_coef(y_true, y_pred):
    y_true_f = keras.backend.flatten(y_true)
    y_pred_f = keras.backend.flatten(y_pred)
    intersection = keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection) / (keras.backend.sum(y_true_f + y_pred_f))

# make the UNET generator model
class inpaintingModel:

    """
        Create a simple UNET model which takes masked images in and outputs created models
    """

    def __init__(self, input_size=(32, 32, 3)):
        self.input_size = input_size

    def prepare_model(self):
        inputs = keras.layers.Input(self.input_size)

        convolution1, pooling1 = self.EncoderBlock(32, (3, 3), (2, 2), 'relu', 'same', inputs)
        convolution2, pooling2 = self.EncoderBlock(64, (3, 3), (2, 2), 'relu', 'same', pooling1)
        convolution3, pooling3 = self.EncoderBlock(128, (3, 3), (2, 2), 'relu', 'same', pooling2)
        convolution4, pooling4 = self.EncoderBlock(256, (3, 3), (2, 2), 'relu', 'same', pooling3)

        convolution5, upscale6 = self.DecoderBlock(512, 256, (3, 3), (2, 2), (2, 2), 'relu', 'same', pooling4, convolution4)
        convolution6, upscale7 = self.DecoderBlock(256, 128, (3, 3), (2, 2), (2, 2), 'relu', 'same', upscale6, convolution3)
        convolution7, upscale8 = self.DecoderBlock(128, 64, (3, 3), (2, 2), (2, 2), 'relu', 'same', upscale7, convolution2)
        convolution8, upscale9 = self.DecoderBlock(64, 32, (3, 3), (2, 2), (2, 2), 'relu', 'same', upscale8, convolution1)

        convolution9 = self.EncoderBlock(32, (3, 3), (2, 2), 'relu', 'same', upscale9, False)

        outputs = keras.layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(convolution9)

        return keras.models.Model(inputs=[inputs], outputs=[outputs])

    def EncoderBlock(self, filters, kernel_size, pool_size, activation, padding, connecting_layer, pool_layer=True):
        conv = keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, activation=activation, padding=padding)(connecting_layer)
        conv = keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, activation=activation, padding=padding)(conv)
        if pool_layer:
            pool = keras.layers.MaxPooling2D(pool_size)(conv)
            return conv, pool
        else:
            return conv

    def DecoderBlock(self, filters, up_filters, kernel_size, up_kernel, up_stride, activation, padding,
                      connecting_layer, shared_layer):
        conv = keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, activation=activation, padding=padding)(connecting_layer)
        conv = keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, activation=activation, padding=padding)(conv)
        up = keras.layers.Conv2DTranspose(filters=up_filters, kernel_size=up_kernel, strides=up_stride, padding=padding)(conv)   # This might be the reason it doesn't work on delegats
        up = keras.layers.concatenate([up, shared_layer], axis=3)

        return conv, up

# Clear the backend
keras.backend.clear_session()

# Create the model, compile it, and train it
model = inpaintingModel(input_size=(DimSize, DimSize, 3)).prepare_model()    # this is the simple UNET
model.summary()
model.compile(optimizer='adam', loss='mean_absolute_error', metrics=[dice_coef])
_ = model.fit(traingen,
              validation_data=testgen,
              epochs=N_epochs,
              steps_per_epoch=len(traingen),
              validation_steps=len(testgen))

if not os.path.exists(pathToSavedModelFold) : os.mkdir(pathToSavedModelFold)
model.save(pathToSavedModelFold + "/generator_model")

# Testing the model and generating some test images
if not os.path.exists(pathToResultsFold): os.mkdir(pathToResultsFold)
[os.remove(pathToResultsFold + f'/{filename}') for filename in os.listdir(pathToResultsFold)]

for sample_idx in range(5):
    sample_images, sample_labels = traingen[sample_idx]
    for i in range(BatchSize):
        impainted_image = model.predict(sample_images[i].reshape((1,) + sample_images[i].shape))
        inpainted_image = impainted_image.reshape(impainted_image.shape[1:])
        inpainted_image = np.asarray(inpainted_image*255, dtype=np.uint8)
        sample_img = np.asarray(sample_images[i]*255, dtype=np.uint8)
        sample_label = np.asarray(sample_labels[i]*255, dtype=np.uint8)

        totImg = np.concatenate([sample_label, sample_img, inpainted_image], axis=1)
        totImg = np.asarray(totImg, dtype=np.uint8)

        cv2.imwrite(f'{pathToResultsFold}/img_{sample_idx}_{i}.png', totImg)

