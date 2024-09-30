# importing the necessary modules
from keras.models import Sequential 
from keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop, SGD
from tensorflow.keras.utils import to_categorical 
from keras.regularizers import l1
import echidna as data
import matplotlib.pyplot as plt
import d_boundary as boundary
import losses

# print('display' in dir(boundary))

# Data pre-processing
X_train = data.X_train
X_validation = data.X_validation
Y_train = to_categorical(data.Y_train)
Y_validation = to_categorical(data.Y_validation)


# Building neural network model
model = Sequential()
model.add(Dense(100, activation='sigmoid', activity_regularizer=l1(0.0004))) 
model.add(Dense(30, activation='sigmoid', activity_regularizer=l1(0.0004)))
model.add(Dense(2, activation='softmax'))

# # Compiling the model
model.compile(loss='categorical_crossentropy', 
              optimizer=RMSprop(learning_rate=0.001),
              metrics=['accuracy'])

# # # Calling the training function for our neural network
history = model.fit(X_train, Y_train,
          validation_data=(X_validation, Y_validation),
          epochs=50000, batch_size=25)

# model2 = Sequential()
# model2.add(Dense(100, activation='sigmoid'))
# model2.add(Dense(2, activation='softmax'))

# model2.compile(loss='categorical_crossentropy', 
#               optimizer=SGD(learning_rate=0.001),
#               metrics=['accuracy'])

# model2.fit(X_train, Y_train,
#           validation_data=(X_validation, Y_validation),
#           epochs=1000, batch_size=25)


# # Display the descision boundary
boundary.show(model, data.X_train, data.Y_train, 'training_reg.png')
boundary.show(model, data.X_validation, data.Y_validation, 'validation_reg.png')

losses.plot(history, 'loss_history_reg.png')


