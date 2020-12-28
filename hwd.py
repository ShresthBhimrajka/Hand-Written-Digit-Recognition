from tensorflow.keras import models
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

def loader():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    
    train_images = train_images/255 
    test_images = test_images/255
    
    train_images = tf.reshape(train_images,[-1,28,28,1])
    test_images = tf.reshape(test_images,[-1,28,28,1])
    
    train_labels = tf.one_hot(train_labels, depth=10)
    test_labels = tf.one_hot(test_labels, depth=10)
    
    return (train_images, train_labels), (test_images, test_labels)

def trainer():
    (train_images, train_labels), (test_images, test_labels) = loader()
    
    model = models.Sequential()
    model.add(Conv2D (32 ,( 5, 5 ), strides = (1, 1), input_shape = (28, 28, 1 ), padding ='valid', activation ='relu', kernel_initializer ='uniform' ))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
        
    model.add(Conv2D (64 ,( 5, 5 ), strides = (1, 1 ), padding ='valid', activation ='relu', kernel_initializer ='uniform'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
        
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))
        
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    
    model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))
    
    predictions = model.predict(test_images)
    test_images = tf.reshape(test_images,[-1,28,28])
    
    tester(test_images, predictions)
    
def tester(test_images, predictions):  
        while True:
            i = int(input("Enter the number to test between 1 and 10000: "))
            print(np.argmax(predictions[i]))
            plt.figure()
            plt.imshow(test_images[i])
            plt.show()
            c = input("Continue [y]/n: ")
            if c == 'n':
                break
            
trainer()