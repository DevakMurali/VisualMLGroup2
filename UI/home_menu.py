from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import QtCore
from PyQt5.uic import loadUi


class HomeMenu(QMainWindow):
    def __init__(self):
        super().__init__()
        super(HomeMenu, self).__init__()
        loadUi("home_menu.ui", self)
        self.setFixedSize(1366, 768)
        self.setStyleSheet("QPushButton "
                                  "{"
                                  "border: none; background: transparent"

                                  "}"
                                  "QPushButton::hover"
                                  "{"
                                  "color : lightblue;"
                                  "}")
