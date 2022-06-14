# -*- coding: utf-8 -*-
"""
Lihi Shapir- Breast cancer project
"""

from tkinter_lihi import Tkinter_Lihi

def main() -> None :
    """
    The main function,
    stars runing the whole project 
    by starting the communication with the user
    """
    tkinter_lihi = Tkinter_Lihi() # contains an object of the class 
    tkinter_lihi.first_menu() # shows to the user the opening screen
    
if __name__== '__main__':
    main()
    
