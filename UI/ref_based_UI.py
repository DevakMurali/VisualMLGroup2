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
from pre_process import alignImages


class RefUI(QMainWindow):
    def __init__(self):
        super().__init__()
        super(RefUI, self).__init__()

        tf.debugging.set_log_device_placement(True)
        loadUi("ref_based_ui.ui", self)
        self.temp_image_path = ""
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

        self.button_template.clicked.connect(lambda: self.load_image(0))
        self.button_test.clicked.connect(lambda: self.load_image(1))
        self.button_detect.clicked.connect(lambda: self.get_defective_image())
        #self.button_optimize.clicked.connect(lambda: self.pre_process())

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

    def get_defective_image(self):
        model = models.load_model("models/reference-based-CNN-weights.hdf5")
        defect_img = generate_prediction_image(str(self.temp_image_path), str(self.test_image_path), model)
        cv.imwrite("generated_images/defects.jpg", defect_img)
        self.graphics_defects.setStyleSheet(
            "border-image: url(generated_images/defects.jpg) 0 0 0 0 stretch stretch;")






