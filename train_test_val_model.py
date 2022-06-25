# -*- coding: utf-8 -*-
"""
Lihi Shapir- Breast cancer project

"""
import os
from tensorflow.keras import Sequential
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout, BatchNormalization
from tensorflow.keras.models import load_model
from tkinter import filedialog
import matplotlib.pyplot as plt
from data_handler import Datahandler
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import cv2


class NN:
    def __init__(self, data_handler):
        self.model = None  # The keras model
        self.last_history = None  # The history of the last training session
        self.data_handler = data_handler  # The data handler


    def train(self) -> None:
        """
        trains the model, shows and shows 2 graphs for the accuracy and the loss of the model
        :return: None
        """

        x_train, y_train = self.data_handler.load_train()
        x_val, y_val = self.data_handler.load_val()
        history = self.model.fit(x_train, y_train, batch_size=32, epochs=20, validation_data=(x_val, y_val))
        self.last_history = history
        loss = self.last_history.history['loss']
        accuracy = self.last_history.history['accuracy']
        val_loss = self.last_history.history['val_loss']
        val_accuracy = self.last_history.history['val_accuracy']

        plt.plot(accuracy)
        plt.plot(val_accuracy)
        plt.title('Model Accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()

        plt.plot(loss)
        plt.plot(val_loss)
        plt.title('Model Loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()

        print("finished training the model choose your next action")
        '''
        fig, (ax1, ax2) = plt.subplots(2)
        ax1.plot(loss, label='loss')
        ax1.plot(accuracy, label='accuracy')
        ax2.plot(val_loss, label='val_loss')
        ax2.plot(val_accuracy, label='val_accuracy')
        plt.show()
        '''

    def test(self) -> None:
        """
        run test on model accuracy and loss
        :return: None
        """
        vis_dir = Datahandler.ask_empty_directory('Choose the visualized test directory')
        num_vis = 100

        x_test, y_test = self.data_handler.load_test()
        result = self.model.evaluate(x_test, y_test, batch_size=32)
        print('test loss: ' + str(result[0]))
        print('test accuracy: ' + str(result[1]))

        idx = np.random.permutation(len(x_test))
        x, y = [x_test[i] for i in idx], [y_test[i] for i in idx]
        x, y = np.array(x[:num_vis]).copy(), np.array(y[:num_vis]).copy()
        preds = self.model.predict(x)

        # a code that visualised 100 photos and tags them as cancer=1 or non cancer=0
        for i in range(num_vis):
            image = x[i] * 255
            image = image.astype(np.uint8)
            image = Image.fromarray(image)
            draw = ImageDraw.Draw(image)
            # Choose a font
            # font = ImageFont.truetype("Roboto-Regular.ttf", 50)
            font = ImageFont.load_default()
            label = np.argmax(y[i])
            pred = np.argmax(preds[i])
            # Draw the text
            draw.text((0, 0), f"{label} as {pred}", font=font, fill='black')
            cv2_im_processed = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(vis_dir, f"result{i}.png"), cv2_im_processed)

    def save_model(self) -> None :
        """
        This function is responsible on saving the trained model,
        saves the h5 file with the name that we got from the user.
        """
        save_name = filedialog.asksaveasfilename(initialdir=os.getcwd(),
                                                 filetypes=(('Keras model File (.h5)', '*.h5'),),
                                                 defaultextension=".h5")
        if save_name != '':
            self.model.save(save_name)
            print('Saved model to: ' + save_name)

    def load_default_model(self) -> None :
        """
        the default model if the user does not want to use 
        the pretrained model
        
        """
        
        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), padding = 'same', activation = 'relu',
                              input_shape = (50, 50, 3)))
        self.model.add(BatchNormalization())
        self.model.add(MaxPool2D(pool_size=(3, 3)))
        self.model.add(Dropout(0.25))
        self.model.add(Conv2D(16, (3, 3), padding="same", activation="relu"))
        self.model.add(MaxPool2D(pool_size=(2, 2)))

        self.model.add(Conv2D(16, (3, 3), padding="same", activation="relu"))
        self.model.add(MaxPool2D(pool_size=(2, 2)))

        self.model.add(Conv2D(32, (3, 3), padding="same", activation="relu"))
        self.model.add(MaxPool2D(pool_size=(2, 2)))
        
        self.model.add(Flatten())
        self.model.add(Dense(2, activation='softmax'))
        adam = keras.optimizers.Adam(learning_rate=0.0001)
        self.model.compile(loss='binary_crossentropy', optimizer= adam, metrics=['accuracy'])
        self.model.summary()

        
    def load_h5_model(self) -> None :
        """
        Loading the pretrained model using the path 
        that the user gave
        """
        model_path = filedialog.askopenfilename(initialdir=os.getcwd(), filetypes=(('Model File', '*.h5'),),
                                                title='Choose the pretrained model h5 file')
        self.model = load_model(model_path)        

    
    