from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import QtCore
from PyQt5.uic import loadUi


class StartMenu(QMainWindow):
    def __init__(self):
        super(StartMenu, self).__init__()
        loadUi("start_menu.ui", self)
        self.setFixedSize(1366, 768)
        self.button.setStyleSheet("QPushButton "
                                  "{"
                                  "border: none; background: transparent"
                    
                                  "}"
                                  "QPushButton::hover"
                                    "{"
                                    "color : lightblue;"
                                    "}")
