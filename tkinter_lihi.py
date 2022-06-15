# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 12:11:26 2022

@author: ARNON
"""

from data_handler import Datahandler
from train_test_val_model import NN
import tkinter as tk

class Tkinter_Lihi:
    def __init__(self):
        self.data_handler = Datahandler() # an object of the class, in order to call diffrent functions in this class. 
        self.nn = NN(self.data_handler) # an object of the class, in order to call diffrent functions in this class. 
        self.window = None # contains the screen that the user will see in each level.

    def first_menu(self) -> None :
        """
        The first menu that will pop up to the user
        this screen is responsible for the data
        """
        self.window = tk.Tk()
        self.window.configure(bg='pink')
        self.window.geometry('500x500')
        tk.Button(self.window, text='Load from zip', command=lambda: [self.data_handler.extract_data(), self.second_menu()],
                  background='#42eff5').pack(anchor='nw', padx=40, pady=50)
        tk.Button(self.window, text='Load from extracted zip',
                  command=lambda: [self.data_handler.delete_corrupted(), self.second_menu()],
                  background='red').pack(anchor='nw', padx=40, pady=50)
        tk.Button(self.window, text='Load from split data',
                  command=lambda: [self.data_handler.choose_split_dir(), self.second_menu()], 
                  background='yellow').pack(anchor='nw', padx=40, pady=50)

        self.window.mainloop()

    def second_menu(self) -> None :
        """
        The second menu that will pop up to the user
        this screen is responsible for the model that we will work with.
        """
        self.window.destroy()

        self.window = tk.Tk()
        self.window.configure(bg='pink')
        self.window.geometry('500x500')
        tk.Button(self.window, text='Load default model',
                  command=lambda: [self.nn.load_default_model(), self.third_menu()]).pack(anchor='nw', padx=40, pady=50)
        tk.Button(self.window, text='Load pretrained model',
                  command=lambda: [self.nn.load_h5_model(), self.third_menu()]).pack(anchor='nw', padx=40, pady=50)
        self.window.mainloop()

    def third_menu(self) -> None :
        """
        The third and last menu that will pop up to the user
        this screen is responsible for training/ testing/ saving the model.
        we can alse return to the prior screen if we want to load a diffrent model.
        """
        self.window.destroy()
        self.window = tk.Tk()
        self.window.configure(bg='pink')
        self.window.geometry('500x500')
        tk.Button(self.window, text='Train model', command=self.nn.train).pack(anchor='nw', padx=40, pady=50)
        tk.Button(self.window, text='Test model', command=self.nn.test).pack(anchor='nw', padx=40, pady=50)
        tk.Button(self.window, text='save trained model', command=self.nn.save_model).pack(anchor='nw', padx=40,
                                                                                           pady=50)
        tk.Button(self.window, text='Back', command=self.second_menu).pack(anchor='nw', padx=40, pady=50)
        self.window.mainloop()
    