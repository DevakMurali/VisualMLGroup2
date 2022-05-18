from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import QtCore
from PyQt5.uic import loadUi
from svm import create_defect_image
import cv2 as cv
from keras import models
import tensorflow as tf
import tkinter as tk
from tkinter import filedialog
from joblib import dump, load
from pre_process import alignImages


class AlignUI(QMainWindow):
    def __init__(self):
        super().__init__()
        super(AlignUI, self).__init__()

        tf.debugging.set_log_device_placement(True)
        loadUi("align_UI.ui", self)
        self.test_image_path = ""
        self.temp_image_path = ""

        self.setFixedSize(1366, 768)
        self.setStyleSheet("QPushButton "
                                  "{"
                                  "border: none; background: transparent"

                                  "}"
                                  "QPushButton::hover"
                                  "{"
                                  "color : lightblue;"
                                  "}")

        self.button_template.clicked.connect(lambda: self.load_image(0))
        self.button_test.clicked.connect(lambda: self.load_image(1))
        self.button_align.clicked.connect(lambda: self.pre_process())

    def load_image(self, path_bool):
        # Code gotten from: https://www.codegrepper.com/code-examples/python/Python+open+file+explorer+to+select+file
        root = tk.Tk()
        root.withdraw()

        file_path = filedialog.askopenfilename(filetypes=[("image", ".jpeg"), ("image", ".jpg"), ("image", ".png")])

        if path_bool == 0:  # Template path
            self.temp_image_path = file_path
        else:  # Test path
            self.test_image_path = file_path

        self.graphics_template.setStyleSheet(
            "border-image: url(" + str(self.temp_image_path) + ") 0 0 0 0 stretch stretch;")
        self.graphics_test.setStyleSheet(
            "border-image: url(" + str(self.test_image_path) + ") 0 0 0 0 stretch stretch;")

    def pre_process(self):
        template_img = cv.imread(str(self.temp_image_path), cv.IMREAD_COLOR)
        test_img = cv.imread(str(self.test_image_path), cv.IMREAD_COLOR)
        aligned_img, _ = alignImages(test_img, template_img)

        cv.imwrite("generated_images/aligned.jpg", aligned_img)
        self.test_image_path = "generated_images/aligned.jpg"
        self.graphics_defects.setStyleSheet(
            "border-image: url(generated_images/aligned.jpg) 0 0 0 0 stretch stretch;")
