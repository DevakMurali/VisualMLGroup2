from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import QtCore
from PyQt5.uic import loadUi
from ref_based import generate_prediction_image
import cv2 as cv
from keras import models
import tensorflow as tf
import tkinter as tk
from tkinter import filedialog


class YoloUI(QMainWindow):
    def __init__(self):
        super().__init__()
        super(YoloUI, self).__init__()

        tf.debugging.set_log_device_placement(True)
        loadUi("yolo_UI.ui", self)
        self.test_image_path = ""

        self.setFixedSize(1366, 768)
        self.setStyleSheet("QPushButton "
                                  "{"
                                  "border: none; background: transparent"

                                  "}"
                                  "QPushButton::hover"
                                  "{"
                                  "color : lightblue;"
                                  "}")

        self.button_test.clicked.connect(lambda: self.load_image())
        #self.button_detect.clicked.connect(lambda: self.get_defective_image())

    def load_image(self):
        # Code gotten from: https://www.codegrepper.com/code-examples/python/Python+open+file+explorer+to+select+file
        root = tk.Tk()
        root.withdraw()

        file_path = filedialog.askopenfilename(filetypes=[("image", ".jpeg"), ("image", ".jpg"), ("image", ".png")])
        self.test_image_path = file_path
        self.graphics_test.setStyleSheet(
            "border-image: url(" + str(self.test_image_path) + ") 0 0 0 0 stretch stretch;")