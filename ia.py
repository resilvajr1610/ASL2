# import data processing and visualisation libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# import image processing libraries
import cv2
import skimage
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix

# import tensorflow and keras
import tensorflow as tf
from tensorflow import keras
import os

print("Packages imported...")

batch_size = 64
imageSize = 64
target_dims = (imageSize, imageSize, 3)
num_classes = 29

train_len = 87000
train_dir = 'asl_alphabet_train/asl_alphabet_train/'

def get_data(folder):
    X = np.empty((train_len, imageSize, imageSize, 3), dtype=np.float32)
    y = np.empty((train_len,), dtype=int)
    cnt = 0
    for folderName in os.listdir(folder):
        if not folderName.startswith('.'):
            if folderName in ['A']:
                label = 0
                print('A');
            elif folderName in ['B']:
                label = 1
                print('B');
            elif folderName in ['C']:
                label = 2
                print('C');
            elif folderName in ['D']:
                label = 3
                print('D');
            elif folderName in ['E']:
                label = 4
                print('E');
            elif folderName in ['F']:
                label = 5
                print('F');
            elif folderName in ['G']:
                label = 6
                print('G');
            elif folderName in ['H']:
                label = 7
                print('H');
            elif folderName in ['I']:
                label = 8
                print('I');
            elif folderName in ['J']:
                label = 9
                print('J');
            elif folderName in ['K']:
                label = 10
                print('K');
            elif folderName in ['L']:
                label = 11
                print('L');
            elif folderName in ['M']:
                label = 12
                print('M');
            elif folderName in ['N']:
                label = 13
                print('N');
            elif folderName in ['O']:
                label = 14
                print('O');
            elif folderName in ['P']:
                label = 15
                print('P');
            elif folderName in ['Q']:
                label = 16
                print('Q');
            elif folderName in ['R']:
                label = 17
                print('R');
            elif folderName in ['S']:
                label = 18
                print('S');
            elif folderName in ['T']:
                label = 19
                print('T');
            elif folderName in ['U']:
                label = 20
                print('U');
            elif folderName in ['V']:
                label = 21
                print('V');
            elif folderName in ['W']:
                label = 22
                print('W');
            elif folderName in ['X']:
                label = 23
                print('X');
            elif folderName in ['Y']:
                label = 24
                print('Y');
            elif folderName in ['Z']:
                label = 25
                print('Z');
            elif folderName in ['del']:
                label = 26
                print('DEL');
            elif folderName in ['nothing']:
                label = 27
                print('NOT');
            elif folderName in ['space']:
                label = 28
                print('SPACE');           
            else:
                label = 29
                print('ELSE');
            for image_filename in os.listdir(folder + folderName):
                img_file = cv2.imread(folder + folderName + '/' + image_filename)
                if img_file is not None:
                    img_file = skimage.transform.resize(img_file, (imageSize, imageSize, 3))
                    img_arr = np.asarray(img_file).reshape((-1, imageSize, imageSize, 3))
                    
                    X[cnt] = img_arr
                    y[cnt] = label
                    cnt += 1
                
    return X,y
X_train, y_train = get_data(train_dir)
print("Images successfully imported...")

print("The shape of X_train is : ", X_train.shape)
print("The shape of y_train is : ", y_train.shape)

print("The shape of one image is : ", X_train[0].shape)

plt.imshow(X_train[0])
plt.show()

X_data = X_train
y_data = y_train
print("Copies made...")


X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3,random_state=42,stratify=y_data)

# One-Hot-Encoding the categorical data
y_cat_train = to_categorical(y_train,29)
y_cat_test = to_categorical(y_test,29)

# Checking the dimensions of all the variables
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
print(y_cat_train.shape)
print(y_cat_test.shape)

# This is done to save CPU and RAM space while working on Kaggle Kernels. This will delete the specified data and save some space!
import gc
del X_data
del y_data
gc.collect()

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dense, Flatten
print("Packages imported...")

model = Sequential()

model.add(Conv2D(32, (5, 5), input_shape=(64, 64, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dense(29, activation='softmax'))

model.summary()

early_stop = EarlyStopping(monitor='val_loss',patience=2)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_cat_train,
          epochs=50,
          batch_size=64,
          verbose=2,
          validation_data=(X_test, y_cat_test),
         callbacks=[early_stop])

metrics = pd.DataFrame(model.history.history)
print("The model metrics are")
metrics

metrics[['loss','val_loss']].plot()
plt.show()

metrics[['accuracy','val_accuracy']].plot()
plt.show()

model.evaluate(X_test,y_cat_test,verbose=0)

predictions = model.predict_classes(X_test)
print("Predictions done...")
print(classification_report(y_test,predictions))

plt.figure(figsize=(12,12))
sns.heatmap(confusion_matrix(y_test,predictions))
plt.show()