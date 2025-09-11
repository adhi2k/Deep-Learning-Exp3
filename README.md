# EX 03 : Convolutional Deep Neural Network for Digit Classification

## AIM

To Develop a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images.

## Problem Statement and Dataset

Digit classification and to verify the response for scanned handwritten images.

The MNIST dataset is a collection of handwritten digits. The task is to classify a given image of a handwritten digit into one of 10 classes representing integer values from 0 to 9, inclusively. The dataset has a collection of 60,000 handwrittend digits of size 28 X 28. Here we build a convolutional neural network model that is able to classify to it's appropriate numerical value.

## Neural Network Model

![Screenshot 2024-09-26 225118](https://github.com/user-attachments/assets/0edcca87-9724-4f6a-ae2e-3ffa8d923a7b)

## DESIGN STEPS
### STEP 1:
Import tensorflow and preprocessing libraries.
### STEP 2:
load the dataset
### STEP 3:
Scale the dataset between it's min and max values
### STEP 4:
Using one hot encode, encode the categorical values
### STEP 5:
Split the data into train and test
### STEP 6:
Build the convolutional neural network model
### STEP 7:
Train the model with the training data
### STEP 8:
Plot the performance plot
### STEP 9:
Evaluate the model with the testing data
### STEP 10:
Fit the model and predict the single input

## PROGRAM
```
Name:Adhithya K
Register Number: 2305002001
```
```
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import utils
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.preprocessing import image
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train.shape
X_test.shape

print("Adhithya K 2305002001")
single_image= X_train[0]
single_image.shape
plt.imshow(single_image,cmap='gray')

y_train.shape
X_train.min()

X_train.max()

X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0

X_train_scaled.min()

X_train_scaled.max()

y_train[0]

y_train_onehot = utils.to_categorical(y_train,10)
y_test_onehot = utils.to_categorical(y_test,10)

 type(y_train_onehot)
y_train_onehot.shape

print("Adhithya K 2305002001")
single_image = X_train[400]
plt.imshow(single_image,cmap='gray')
y_train_onehot[500]
X_train_scaled = X_train_scaled.reshape(-1,28,28,1)
X_test_scaled = X_test_scaled.reshape(-1,28,28,1)

model = keras.Sequential()
model.add(layers.Input(shape=(28,28,1)))
model.add(layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu'))
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(32,activation='relu'))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train_scaled ,y_train_onehot, epochs=5,
          batch_size=128,
          validation_data=(X_test_scaled,y_test_onehot))
metrics = pd.DataFrame(model.history.history)

metrics.head()

print("Adhithya K 2305002001")
metrics = pd.DataFrame(model.history.history)
metrics.head()

print("Adhithya K 2305002001")
metrics[['accuracy','val_accuracy']].plot()
print("Adhithya K 2305002001")
metrics[['loss','val_loss']].plot()
print("Adhithya K 2305002001")
x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)
print("Adhithya K 2305002001")
print(confusion_matrix(y_test,x_test_predictions))
print("Adhithya K 2305002001")
print(classification_report(y_test,x_test_predictions))

img = image.load_img('/content/imgs.png')
img_tensor = tf.convert_to_tensor(np.asarray(img))
img_28 = tf.image.resize(img_tensor,(28,28))
img_28_gray = tf.image.rgb_to_grayscale(img_28)
img_28_gray_scaled = img_28_gray.numpy()/255.0
x_single_prediction = np.argmax(
    model.predict(img_28_gray_scaled.reshape(1,28,28,1)),
     axis=1)

print(x_single_prediction)
print("Adhithya K 2305002001")
plt.imshow(img_28_gray_scaled.reshape(28,28),cmap='gray')
img_28_gray_inverted = 255.0-img_28_gray
img_28_gray_inverted_scaled = img_28_gray_inverted.numpy()/255.0
x_single_prediction = np.argmax(
    model.predict(img_28_gray_inverted_scaled.reshape(1,28,28,1)),
     axis=1)
print("Adhithya K 2305002001")
print(x_single_prediction)


```



## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot
!<img width="657" height="527" alt="image" src="https://github.com/user-attachments/assets/d284d493-a4fb-4481-9450-0a79491e1128" />



![OP32](<img width="663" height="531" alt="image" src="https://github.com/user-attachments/assets/6eaa0f25-d75d-4f39-bf24-17537887e5a9" />
)

### Classification Report

![OP33](<img width="519" height="362" alt="image" src="https://github.com/user-attachments/assets/a3e7853a-eadc-4607-8760-37eaf40414ba" />
)


### Confusion Matrix

![OP34](<img width="510" height="238" alt="image" src="https://github.com/user-attachments/assets/5afc5ad2-83f9-4d10-b0a1-abfe9c5d108b" />
)

### New Sample Data Prediction

##### Input

![op35 (2)](<img width="513" height="522" alt="image" src="https://github.com/user-attachments/assets/4630310d-a944-4ab4-898c-2fb539ab64f4" />
)

##### Output
![op36 (2)](<img width="435" height="117" alt="image" src="https://github.com/user-attachments/assets/154a302c-c53e-4949-80f9-a69c7b6faf44" />
)

## RESULT
  Thus, a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images is written and executed successfully.
