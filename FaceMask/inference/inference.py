from PySide6.QtWidgets import *
from PySide6.QtUiTools import *
from PySide6.QtCore import *
from PySide6.QtGui import *
import cv2
import numpy as np
from model import recognizer


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        loader = QUiLoader()
        self.ui = loader.load('Ui/main.ui', None)
        self.ui.show()
        self.rc = recognizer()
        self.camera()


    def camera(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            predict = self.rc.recognize(frame_rgb)
            self.ui.predict_label.setText(predict)
            frame_rgb = cv2.resize(frame_rgb, (600,400))
            #cv2.putText(frame_rgb, predict, (0, 350), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 2)
            img = QImage(frame_rgb, frame_rgb.shape[1], frame_rgb.shape[0], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(img)
            self.ui.camera_label.setPixmap(pixmap)
           

            if cv2.waitKey(1) == ord("q"):
                break

        cap.release()
        cv2.destroyWindow('Mask Detector')







if __name__=='__main__':
    app = QApplication([])
    window = MainWindow()
    app.exec()