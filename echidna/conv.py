# A convolutional neural network that trains on CIFAR-10 images.

import numpy as np
import tensorflow as tf 
from keras.models import Sequential
from keras.layers import Conv2D, Dropout, Dense
from keras.layers import BatchNormalization, Flatten
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.utils import to_categorical
from keras.datasets import cifar10
import losses

# Set a random seed for reproducibility
tf.random.set_seed(42)

# Downloading and pre-processing dataset
(X_train_raw, Y_train_raw), (X_test_raw, Y_test_raw) = cifar10.load_data()

# Selecting a subset of the actual dataset to save training time
(X_train_raw, Y_train_raw) =  (X_train_raw[0:5000], Y_train_raw[0:5000])
(X_test_raw, Y_test_raw) = (X_test_raw[0:500], Y_test_raw[0:500])

X_train = X_train_raw / 255
X_test_all = X_test_raw / 255
X_validation, X_test = np.split(X_test_all, 2)
Y_train = to_categorical(Y_train_raw)
Y_validation, Y_test = np.split(to_categorical(Y_test_raw), 2)


def create_model():
    # Building a convolutional neural network from scratch
    model = Sequential()

    model.add(Conv2D(16, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Flatten())

    model.add(Dense(5000, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(Dense(2000, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(Dense(1000, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))


    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))


    model.add(Dense(10, activation='softmax'))


    # Compiling the model
    model.compile(loss='categorical_crossentropy',
                optimizer=Adam(),
                metrics=['accuracy'])
    return model

# # Training phase for CNN model

model = create_model()
history512 = model.fit(X_train, Y_train,
                    validation_data=(X_validation, Y_validation),
                    epochs=50, batch_size=512)
losses.plot(history512, 'loss_history_reg_conv512.png')
# 


model = create_model()
# Training phase for CNN model
history256 = model.fit(X_train, Y_train,
                    validation_data=(X_validation, Y_validation),
                    epochs=50, batch_size=256)
losses.plot(history256, 'loss_history_reg_conv256.png')


model = create_model()
# Training phase for CNN model
history128 = model.fit(X_train, Y_train,
                    validation_data=(X_validation, Y_validation),
                    epochs=50, batch_size=128)
losses.plot(history128, 'loss_history_reg_conv128.png')


model = create_model()
# Training phase for CNN model
history64 = model.fit(X_train, Y_train,
                    validation_data=(X_validation, Y_validation),
                    epochs=50, batch_size=64)
losses.plot(history64, 'loss_history_reg_conv64.png')


histories = {
    'history512':history512,
    'history256':history256,
    'history128':history128,
    'history64':history64
}

losses.plot_histories(histories, 'loss_history_reg_conv.png')
# losses.plot(history512, 'loss_history_reg_conv512.png')
# losses.plot(history256, 'loss_history_reg_conv256.png')
# losses.plot(history128, 'loss_history_reg_conv128.png')
# losses.plot(history64, 'loss_history_reg_conv64.png')
