# A lot of code from: https://www.youtube.com/watch?v=RYdAf2NH0TY
from PyQt5.QtWidgets import *
from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5 import *

import sys
import os

from home_menu import HomeMenu
from start_menu import StartMenu
from ref_based_UI import RefUI
from yolo_UI import YoloUI
from svm_UI import SVMUI
from align_UI import AlignUI

if __name__ == '__main__':
    # Clear generated_images file (gotten from) https://www.techiedelight.com/delete-all-files-directory-python/

    direct = 'generated_images'
    for f in os.listdir(direct):
        os.remove(os.path.join(direct, f))

    # Lo
    app = QApplication(sys.argv)
    widget = QtWidgets.QStackedWidget()
    widget.setFixedSize(1366, 768)
    app.setStyleSheet('''
            QWidget {
                background-color: white;
            }
    ''')

    start_menu = StartMenu()
    home_menu = HomeMenu()
    ref = RefUI()
    yolo = YoloUI()
    svm = SVMUI()
    align = AlignUI()

    widget.addWidget(start_menu)
    widget.addWidget(home_menu)
    widget.addWidget(ref)
    widget.addWidget(yolo)
    widget.addWidget(svm)
    widget.addWidget(align)

    def increment_stack_index(amount):
        widget.setCurrentIndex(widget.currentIndex() + amount)

    def decrement_stack_index(amount):
        widget.setCurrentIndex(widget.currentIndex() - amount)

    start_menu.button.clicked.connect(lambda: increment_stack_index(1))
    home_menu.button_ref.clicked.connect(lambda: increment_stack_index(1))
    home_menu.button_yolo.clicked.connect(lambda: increment_stack_index(2))
    home_menu.button_svm.clicked.connect(lambda: increment_stack_index(3))
    home_menu.button_align.clicked.connect(lambda: increment_stack_index(4))

    ref.button_back.clicked.connect(lambda: decrement_stack_index(1))
    yolo.button_back.clicked.connect(lambda: decrement_stack_index(2))
    svm.button_back.clicked.connect(lambda: decrement_stack_index(3))
    align.button_back.clicked.connect(lambda: decrement_stack_index(4))
    widget.show()

    try:
        sys.exit(app.exec_())
    except SystemExit:
        print('Closing Window...')


