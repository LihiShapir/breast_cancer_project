from data_handler import Datahandler
from train_test_val_model import NN
import tkinter as tk


class Tkinter_Lihi:
    def __init__(self):
        self.data_handler = Datahandler()
        self.nn = NN(self.data_handler)

        self.window = None

    def first_menu(self):
        self.window = tk.Tk()
        self.window.configure(bg='pink')
        self.window.geometry('500x500')
        tk.Button(self.window, text='Load from zip', command=lambda: [self.data_handler.extract_data(), self.second_menu()],
                  background='#42eff5').pack(anchor='nw', padx=40, pady=50)
        tk.Button(self.window, text='Load from extracted zip',
                  command=lambda: [self.data_handler.delete_corrupted(), self.second_menu()]).pack(anchor='nw', padx=40,
                                                                                               pady=50)
        tk.Button(self.window, text='Load from fixed data',
                  command=lambda: [self.data_handler.split_data(), self.second_menu()]).pack(anchor='nw', padx=40, pady=50)
        tk.Button(self.window, text='Load from split data',
                  command=lambda: [self.data_handler.choose_split_dir(), self.second_menu()]).pack(anchor='nw', padx=40,
                                                                                               pady=50)

        self.window.mainloop()

    def second_menu(self):
        self.window.destroy()

        self.window = tk.Tk()
        self.window.configure(bg='pink')
        self.window.geometry('500x500')
        tk.Button(self.window, text='Load default model',
                  command=lambda: [self.nn.load_default_model(), self.third_menu()]).pack(anchor='nw', padx=40, pady=50)
        tk.Button(self.window, text='Load pretrained model',
                  command=lambda: [self.nn.load_h5_model(), self.third_menu()]).pack(anchor='nw', padx=40, pady=50)
        self.window.mainloop()

    def third_menu(self):
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
