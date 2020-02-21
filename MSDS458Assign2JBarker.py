# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 21:43:40 2020

@author: jbark
"""

import numpy as np

from keras.datasets import cifar10
from keras.utils.np_utils import to_categorical   
import tensorflow as tf
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import Adam
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

#Lists to hold model data
Layers = []
LearnRate = []
Drop = []
Processing_Time = []
Training_Set_Accuracy = []
Test_Set_Accuracy = []
History = []
Predicts = []
Classes = []

#Load CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

print("Shape of training data:")
print(X_train.shape)
print(y_train.shape)
print("Shape of test data:")
print(X_test.shape)
print(y_test.shape)

#Create early stopping mechanisim
max_epochs = 100  
earlystop_callback = \
    tf.keras.callbacks.EarlyStopping(monitor='val_acc',\
    min_delta=0.01, patience=5, verbose=0, mode='auto',\
    baseline=None, restore_best_weights=False)

#Build a variety of models 1 to 3 layers, dropout Y or N and Adam Learning Rate
for layers in range(1,4):
    for learnrate in [0.01, 0.001, 0.002]:
        for drop in range(0,2):
            model = Sequential()

            model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
            model.add(Conv2D(32, (3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
        
            if(layers > 1):
                model.add(Conv2D(64, (3, 3), activation='relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))

            if(layers > 2):
                model.add(Conv2D(128, (3, 3), activation='relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))

            if(drop == 1):
                model.add(Dropout(0.3))

            model.add(Flatten())
            model.add(Dense(256, activation='relu'))
            model.add(Dense(10, activation='softmax'))
        
            model.summary()

            adam = Adam(lr=learnrate)
            model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=adam)

            start = time.perf_counter()
            history = model.fit(X_train, y_train,
                epochs = max_epochs, shuffle = False,
                batch_size=32, validation_split = 0.2, 
                verbose = 2,
                callbacks = [earlystop_callback])

            elapsed = time.perf_counter() - start
            y_preds = model.predict(X_test, verbose=0)
            y_classes = model.predict_classes(X_test, verbose=0)
        
            Layers.append(layers)
            LearnRate.append(learnrate)
            Drop.append(drop)
            Processing_Time.append(elapsed)
            score, acc_train = model.evaluate(X_train, y_train)
            score, acc_test = model.evaluate(X_test, y_test)
            Training_Set_Accuracy.append(acc_train)
            Test_Set_Accuracy.append(acc_test)
            History.append(history)
            Predicts.append(y_preds)            
            Classes.append(y_classes)
            
#Method to print loss chart
def plotLosseslr(history, layers, lr, drop):  
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    dropoutuse = "dropout of rate .3"

    if(drop == 0):
        dropoutuse = "no dropout"

    plt.title('Model Loss for a Convnet with ' + str(layers) + 
              ' layer and a learning rate of ' + str(lr) + ' and ' + dropoutuse)

    if(layers > 1):
        plt.title('Model Loss for a Convnet with ' + str(layers) + 
                  ' layer and a learning rate of ' + str(lr) + ' and ' + dropoutuse)

    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

#Code to print perfomance table loss chart and metric data
from prettytable import PrettyTable
table = PrettyTable(['Layers', 'Learning_Rate', 'Dropout', 'Process_Time', 
                     'Train_Set_Accuracy', 'Validation_Set_Accuracy'])
for x in range(0, 18):
    table.add_row([Layers[x], LearnRate[x], Drop[x],
                   round(Processing_Time[x], 3), 
                   round(Training_Set_Accuracy[x], 3), 
                   round(Test_Set_Accuracy[x], 3)])
print(table)

for x in range(0, 18):
    plotLosseslr(History[x], Layers[x], LearnRate[x], Drop[x])
    y_labels = np.argmax(y_test, axis = 1)
    test_preds = np.argmax(Predicts[x], axis = 1)
    print("Test set F1 (weighted average):", \
       metrics.precision_recall_fscore_support(y_labels, test_preds, average='weighted')[2], "\n")
    print("\nClassification report:\n%s"
          % (metrics.classification_report(y_labels, test_preds, \
    	target_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])))
    cm_data = metrics.confusion_matrix(y_labels, test_preds)
    print('Confusion matrix')
    print('(rows = true digits, columns = predicted digits)\n%s' % cm_data)

