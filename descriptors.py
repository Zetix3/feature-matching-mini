from abc import ABC, abstractmethod
import cv2 as cv
from matchers import Matchers

class BaseDescriptors(ABC):
    @abstractmethod
    def detect_and_compute(self, img1, img2):
        pass


class SIFT(BaseDescriptors):
    def detect_and_compute(self, img1, img2):
        sift = cv.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)
        return kp1, des1, kp2, des2

class ORB(BaseDescriptors):
    def detect_and_compute(self, img1, img2):
         orb = cv.ORB_create()
         kp1, des1 = orb.detectAndCompute(img1, None)
         kp2, des2 = orb.detectAndCompute(img2, None)
         return kp1, des1, kp2, des2

class Descriptors:
    _descriptors = {
        'sift': SIFT,
        'orb': ORB
    }

    def select_descriptors(self, method):
        method_class = self._descriptors.get(method.lower())
        if not method_class:
            raise ValueError(f"Method '{method}' not found ")
        return method_class()

    def apply_method(self, img1, img2, method):
        method_instance = self.select_descriptors(method)
        kp1, des1, kp2, des2 = method_instance.detect_and_compute(img1, img2)
        return kp1, des1, kp2, des2