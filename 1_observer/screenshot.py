import numpy as np
import cv2 as cv
import pyautogui


class Screenshot:
	"""
	Take a screenshot
	Get the height and width
	Preprocess the image into a form better suited for image matching: grayscale + threshold + blur
	"""

	def __init__(self):
		self.bgr = np.array(pyautogui.screenshot())[:, :, ::-1]
		self.h, self.w = self.bgr.shape[0:2]
		self.gray = self.process_gray()

	def process_gray(self):
		return cv.blur(
			cv.threshold(
				cv.cvtColor(self.bgr, cv.COLOR_BGR2GRAY),
				253, 255, cv.THRESH_BINARY)[1],
			(6, 6)
		)