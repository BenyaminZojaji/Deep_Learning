import cv2
import numpy as np
import tensorflow as tf

class recognizer:
    def __init__(self):
        self.model = tf.keras.models.load_model('model/MobileNetV2-faceMask.h5')

    def recognize(self, img_rgb):
        width = height = 224
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img_rgb, (width, height))
        img = img / 255
        img = img.reshape(1, width, height, 3) # 3d -> 4d

        devices = ['With Mask 😷✅','Without Mask 😒⛔']
        result = np.argmax(self.model.predict(img))
        return devices[result]