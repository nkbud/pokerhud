import os

import time
import numpy as np
import cv2 as cv

from screenshot import Screenshot

"""
Idea 1: 15 minutes? --> Less time
- Currently hardcoded to take (100 screenshots) over (15 minutes)
- Could opt for a more dynamic approach where we stop when we think we've got 'em all

"""

class CardLearner:
	"""
	It takes screenshots for a while
	And then it produces a set of images that are hopefully cards
	"""

	def __init__(self, screenshot_limit=100, sleep_interval_seconds=9, basedir="./data"):
		self.card_count = 0
		self.screenshot_limit = screenshot_limit
		self.sleep_interval_seconds = sleep_interval_seconds
		# some locally expected directory structure
		self.basedir = basedir
		self.collectdir = os.path.join(self.basedir, "collect")
		self.clusterdir = os.path.join(self.basedir, "cluster")
		os.makedirs(self.collectdir, exist_ok=True)
		os.makedirs(self.clusterdir, exist_ok=True)

	def execute(self):
		self.collect()
		self.cluster()
		self.combine()

	def collect(self):
		"""
		Takes a bunch of screenshots
		"""

		# announce what's about to happen...
		expected_seconds = self.screenshot_limit * self.sleep_interval_seconds
		expected_minutes = expected_seconds // 60
		remainder_seconds = expected_seconds % 60
		print(f"For an estimated {expected_minutes} minutes and {remainder_seconds} seconds...")
		warning_seconds = 5
		for i in range(warning_seconds):
			time.sleep(1)
			print(f"Taking {self.screenshot_limit} screenshots... {warning_seconds - i}")

		# for each screenshot
		for screenshot_count in range(self.screenshot_limit):

			# pre-process into grayscale, and extract the contours
			screenshot = Screenshot()
			contours = cv.findContours(screenshot.gray, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[0]

			# for each contour
			cardscores = []
			for contour in contours:

				# examine the properties of the bounding rectangle
				bounding_rect = cv.boundingRect(contour)
				x = bounding_rect[0]
				y = bounding_rect[1]
				w = bounding_rect[2]
				h = bounding_rect[3]
				ratio = h / w

				# assert some properties to filter the false-positive contours
				taller_than_wide = 1 < ratio < 1.6
				not_too_tall = h / float(screenshot.h) < 0.1
				not_too_wide = w / float(screenshot.w) < 0.1
				card_like = taller_than_wide and not_too_tall and not_too_wide
				# ... more 'card-like' properties? we can do some statistics
				# ... tbd.

				# save this bounding rect + contour's info / "score" so we can use group stats to apply filters
				cardscores.append(0 if not card_like else h)

			# save only the best X %, X = 5
			max_score = np.amax(cardscores)
			scores_scaled = [float(score) / max_score for score in cardscores]
			for i in range(len(contours)):
				if scores_scaled[i] >= 0.95:
					bounded_image = screenshot.gray[y:(y+h), x:(x+w)]
					self.card_count += 1
					print(f"Found card # {self.card_count}")
					cv.imwrite(f"{self.collectdir}/{self.card_count}.png", bounded_image)

			# report progress, repeat
			print(f"Completed screenshot # {screenshot_count} of {self.screenshot_limit}")
			print(f"Sleeping {self.sleep_interval_seconds} seconds...")
			time.sleep(self.sleep_interval_seconds)

	@staticmethod
	def pad(img, pad_h, pad_w):
		h, w = img.shape[0:2]
		return cv.copyMakeBorder(img, 0, pad_h - h, 0, pad_w - w, cv.BORDER_CONSTANT, value=0)

	@staticmethod
	def match(img1, img2):
		return np.amin(cv.matchTemplate(img1, img2, cv.TM_SQDIFF_NORMED)[0]) < 0.05

	@staticmethod
	def process_imgdir(imgdir):
		# read every file as a grayscale image
		imgs = [
			cv.cvtColor(cv.imread(os.path.join(imgdir, entry)), cv.COLOR_BGR2GRAY)
			for entry in os.listdir(imgdir)
			if os.path.isfile(os.path.join(imgdir, entry))
		]
		# pad zeros to equal height and width
		max_h = np.amax([img.shape[0:2][0] for img in imgs])
		max_w = np.amax([img.shape[0:2][1] for img in imgs])
		return [CardLearner.pad(img, max_h, max_w) for img in imgs]

	def cluster(self):
		"""
		By applying a similarity function on cards A and B, we can decide if A and B should be grouped
		"""
		print(f"Collection complete. Clustering found cards by similarity.")

		# read every file as a grayscale image
		imgs = CardLearner.process_imgdir(self.collectdir)

		# assign each image id to a group, image 0 --> group 0
		id2group = {0: 0}
		num_groups = len(id2group)

		# images --> end, TBD
		for i in range(1, len(imgs)):
			ungrouped = imgs[i]

			# look for a match and save it if so
			found = False
			for id, group in id2group.items():
				grouped = imgs[id]
				if CardLearner.match(ungrouped, grouped):
					id2group[i] = group
					found = True
					break
			if found:
				continue

			# else this img just started a new group
			id2group[i] = num_groups
			num_groups += 1

		# put imgs in their group's folder
		for id, group in id2group.items():
			os.makedirs(f"{self.clusterdir}/{group}", exist_ok=True)
			cv.imwrite(f"{self.clusterdir}/{group}/{id}.png", imgs[id])


	def combine(self):
		"""
		Combine all images in a group into a single average
		"""
		print("Clustering complete. Aggregating clusters to a single representative")

		# for each groupdir
		for d in os.listdir(self.clusterdir):
			groupdir = f"{self.clusterdir}/{d}"
			if not os.path.isdir(groupdir):
				continue

			# get all grayscaled + padded imgs
			imgs = CardLearner.process_imgdir(groupdir)

			# get average
			blank_canvas = np.zeros(imgs[0].shape[0:2], float)
			for img in imgs:
				cv.accumulate(img, blank_canvas)
			canvas = (blank_canvas / len(imgs)).astype(np.uint8)

			# and write the img as a file to outputs
			contour = cv.findContours(canvas, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[0][0]
			x, y, w, h = cv.boundingRect(contour)
			found_box = canvas[y:(y + h), x:(x + w)]
			cv.imwrite(f"{self.clusterdir}/_{d}.png", found_box)

















