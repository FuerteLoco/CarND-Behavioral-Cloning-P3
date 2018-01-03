import csv
import cv2
import sklearn
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.utils.visualize_util import plot
from sklearn.model_selection import train_test_split

# set some parameters
nb_img_per_sample = 6       # 6 images per sample (center, left, right and all flipped)
nb_epoch = 4                # number of epochs
steering_correction = 0.2   # steering correction for left and right image
dropout_ratio = 0.3         # ratio for dropout layer

# read driving log
samples = []
with open("data/driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

# generator for providing batches of images
def generator(samples, batch_size=36):
    num_samples = len(samples)
    while 1:
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size//nb_img_per_sample):
            batch_samples = samples[offset:offset+batch_size//nb_img_per_sample]
            images = []
            steerings = []

            for batch_sample in batch_samples:
                # read images and convert from BGR to RGB
                img_center = cv2.imread(batch_sample[0])[...,::-1]
                img_left = cv2.imread(batch_sample[1])[...,::-1]
                img_right = cv2.imread(batch_sample[2])[...,::-1]

                # get center steering angles and caluclate angels for left and right view
                steering_center = float(batch_sample[3])
                steering_left = steering_center + steering_correction
                steering_right = steering_center - steering_correction

                # add images and angles to data set
                images.append(img_center)
                images.append(img_left)
                images.append(img_right)
                steerings.append(steering_center)
                steerings.append(steering_left)
                steerings.append(steering_right)

                # add flipped images and angles to data set
                images.append(np.fliplr(img_center))
                images.append(np.fliplr(img_left))
                images.append(np.fliplr(img_right))
                steerings.append(-steering_center)
                steerings.append(-steering_left)
                steerings.append(-steering_right)

            X_train = np.array(images)
            y_train = np.array(steerings)
            yield sklearn.utils.shuffle(X_train, y_train)

# get image shape from first image
image_shape = cv2.imread(samples[0][0]).shape

# setup training and validation samples
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
train_generator = generator(train_samples, batch_size=36)
validation_generator = generator(validation_samples, batch_size=36)

# create the model
model=Sequential()
model.add(Lambda(lambda x: x/255.0-0.5, input_shape=image_shape))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Flatten())
model.add(Dropout(dropout_ratio))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

# plot the model
plot(model, to_file="model.png", show_shapes=True)

# compile and train the model using the generator function
model.compile(loss="mse", optimizer="adam")
history_object = model.fit_generator(train_generator,
                                     samples_per_epoch=nb_img_per_sample*len(train_samples),
                                     validation_data=validation_generator,
                                     nb_val_samples=nb_img_per_sample*len(validation_samples),
                                     nb_epoch=nb_epoch)
# save the model
model.save("model.h5")

# plot the training and validation loss for each epoch
x_ticks = range(1, nb_epoch+1)
plt.plot(x_ticks, history_object.history['loss'], "o-")
plt.plot(x_ticks, history_object.history['val_loss'], "o-")
plt.title('Model mean squared error loss')
plt.ylabel('Mean squared error loss')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Validation set'], loc='upper right')
plt.xticks(x_ticks)
plt.savefig("loss.png", bbox_inches="tight")
