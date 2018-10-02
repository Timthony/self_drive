#import pandas as pd # data analysis toolkit - create, read, update, delete datasets
import numpy as np #matrix math
from sklearn.model_selection import train_test_split #to split out training and testing data 
#keras is a high level wrapper on top of tensorflow (machine learning library)
#The Sequential container is a linear stack of layers
import tensorflow
from keras.models import Sequential
#popular optimization strategy that uses gradient descent 
from keras.optimizers import Adam,SGD
#to save our model periodically as checkpoints for loading later
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard
#what types of layers do we want our model to have?
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from keras.models import Model, Input
from keras.models import load_model
import glob
import os
import glob
import h5py
import sys
import keras
#for debugging, allows for reproducible (deterministic) results 
np.random.seed(0)

IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 120, 160, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

def load_data():
    """
    Load training data and split it into training and validation set
    """

    # load training data
    image_array = np.zeros((1,120,160,3))
    label_array = np.zeros((1, 5), 'float')
    training_data = glob.glob('training_data_npz/*.npz')

    # if no data, exit
    if not training_data:
        print ("No training data in directory, exit")
        sys.exit()

    for single_npz in training_data:
        with np.load(single_npz) as data:
            print(data.keys())
            train_temp = data['train_imgs']
            train_labels_temp = data['train_labels']
        image_array = np.vstack((image_array, train_temp))
        label_array = np.vstack((label_array, train_labels_temp))
    
    X = image_array[1:, :]
    y = label_array[1:, :]
    print ('Image array shape: '+ str(X.shape))
    print ('Label array shape: '+ str(y.shape))
    print(np.mean(X))
    print(np.var(X))

    #now we can split the data into a training (80), testing(20), and validation set
    #thanks scikit learn
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=0)

    return X_train, X_valid, y_train, y_valid


def build_model(keep_prob):
    """
    NVIDIA model used
    Image normalization to avoid saturation and make gradients work better.
    Convolution: 5x5, filter: 24, strides: 2x2, activation: ELU
    Convolution: 5x5, filter: 36, strides: 2x2, activation: ELU
    Convolution: 5x5, filter: 48, strides: 2x2, activation: ELU
    Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
    Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
    Drop out (0.5)
    Fully connected: neurons: 100, activation: ELU
    Fully connected: neurons: 50, activation: ELU
    Fully connected: neurons: 10, activation: ELU
    Fully connected: neurons: 1 (output)

    # the convolution layers are meant to handle feature engineering
    the fully connected layer for predicting the steering angle.
    dropout avoids overfitting
    ELU(Exponential linear unit) function takes care of the Vanishing gradient problem. 
    """
    # IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 240, 240, 3
    # INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)
    
    model = Sequential()
    model.add(Lambda(lambda x: (x/102.83-1), input_shape=INPUT_SHAPE))#/85-5139
    model.add(Conv2D(24, (5, 5), activation='elu', strides=(2, 2)))
    model.add(Conv2D(36, (5, 5), activation='elu', strides=(2, 2)))
    model.add(Conv2D(48, (5, 5), activation='elu', strides=(2, 2)))

    #model.add(Dropout(0.5))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    #model.add(Dropout(0.3))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Dropout(keep_prob))
    model.add(Flatten())
    model.add(Dense(500, activation='elu'))
    #model.add(Dropout(0.1))
    model.add(Dense(250, activation='elu'))
    #model.add(Dropout(0.1))
    model.add(Dense(50, activation='elu'))
    #model.add(Dropout(0.1))
    model.add(Dense(5))
    model.summary()

    return model


def train_model(model,learning_rate,nb_epoch,samples_per_epoch, batch_size,X_train, X_valid, y_train, y_valid):
    """
    Train the model
    """
    #Saves the model after every epoch.
    #quantity to monitor, verbosity i.e logging mode (0 or 1), 
    #if save_best_only is true the latest best model according to the quantity monitored will not be overwritten.
    #mode: one of {auto, min, max}. If save_best_only=True, the decision to overwrite the current save file is
    # made based on either the maximization or the minimization of the monitored quantity. For val_acc, 
    #this should be max, for val_loss this should be min, etc. In auto mode, the direction is automatically
    # inferred from the name of the monitored quantity.
    checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=True,
                                 mode='min')

    # EarlyStopping patience：当early 
    # stop被激活（如发现loss相比上一个epoch训练没有下降），则经过patience个epoch后停止训练。 
    # mode：‘auto’，‘min’，‘max’之一，在min模式下，如果检测值停止下降则中止训练。在max模式下，当检测值不再上升则停止训练。
    early_stop = EarlyStopping(monitor='val_loss', min_delta=.0005, patience=4, 
                                         verbose=1, mode='min')
    tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=20, write_graph=True, write_grads=True, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
    #calculate the difference between expected steering angle and actual steering angle
    #square the difference
    #add up all those differences for as many data points as we have
    #divide by the number of them
    #that value is our mean squared error! this is what we want to minimize via
    #gradient descent
    #opt = SGD(lr=0.0001)
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=learning_rate),metrics=['accuracy'])#

    #Fits the model on data generated batch-by-batch by a Python generator.

    #The generator is run in parallel to the model, for efficiency. 
    #For instance, this allows you to do real-time data augmentation on images on CPU in 
    #parallel to training your model on GPU.
    #so we reshape our data into their appropriate batches and train our model simulatenously

    model.fit_generator(batch_generator(X_train, y_train,batch_size),
                    steps_per_epoch=samples_per_epoch/batch_size,
                    epochs=nb_epoch,
                    max_queue_size=1,
                    validation_data=batch_generator(X_valid, y_valid, batch_size),
                    validation_steps=len(X_valid)/batch_size,
                    callbacks=[tensorboard, checkpoint, early_stop],
                    verbose=2)
    
##    model.fit(X_train,y_train,samples_per_epoch,nb_epoch,max_q_size=1,X_valid,y_valid,\
##              nb_val_samples=len(X_valid),callbacks=[checkpoint],verbose=1)

def batch_generator(X, y, batch_size):
    """
    Generate training image give image paths and associated steering angles
    """
    images = np.empty([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
    steers = np.empty([batch_size,5])
    while True:
        i = 0
        for index in np.random.permutation(X.shape[0]):
            images[i] = X[index]
            steers[i] = y[index]
            i += 1
            if i == batch_size:
                break
        yield (images, steers)

def main():
    #print parameters
    print('-' * 30)
    print('Parameters')
    print('-' * 30)
    
    keep_prob = 0.5
    learning_rate = 0.0001
    nb_epoch = 100
    samples_per_epoch = 3000
    batch_size = 40

    print('keep_prob = %f', keep_prob)
    print('learning_rate = %f', learning_rate)
    print('nb_epoch = %d', nb_epoch)
    print('samples_per_epoch = %d', samples_per_epoch)
    print('batch_size = %d', batch_size)
    print('-' * 30)

    #load data
    data = load_data()
    #build model
    model = build_model(keep_prob)
    #train model on data, it saves as model.h5 
    train_model(model,learning_rate,nb_epoch,samples_per_epoch, batch_size, *data)


if __name__ == '__main__':
    main()
