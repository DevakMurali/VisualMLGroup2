import numpy as np
import cv2
import cv2 as cv
import copy
import csv
import requests
from sklearn import svm
import random


def generate_images(defect_list, contours, color, template_image, IMAGE_WIDTH=100, IMAGE_HEIGHT=100):

  centroids = []
  context_defects = []

  outside_image_width = len(template_image[0])
  outside_image_height = len(template_image[1])

  black_bg = np.zeros((outside_image_height, outside_image_width, 3))

  for defect in defect_list:
    context_image = template_image.copy()

    total_centroid = np.asarray((0.0, 0.0))

    for centroid in defect[1]:
      total_centroid += centroid

    for idx in defect[0]:
      cv.drawContours(context_image, contours, idx, color, -1)

    total_centroid /= len(defect[1])
    cnt = defect[2]

    x,y,w,h = cv2.boundingRect(cnt)

    image_bound = max(w, h) + 10

    l_side_x = int(total_centroid[0] - image_bound / 2)
    r_side_x = int(total_centroid[0] + image_bound / 2)
    top_side_y = int(total_centroid[1] - image_bound / 2)
    bottom_side_y = int(total_centroid[1] + image_bound / 2)

    template_unscaled = template_image[top_side_y:bottom_side_y, l_side_x:r_side_x]
    (x, y, z) = np.shape(template_unscaled)

    if x > 40 and y > 40:
      centroids.append(total_centroid)
      context_defects.append(cv2.resize(template_unscaled, (IMAGE_WIDTH, IMAGE_HEIGHT)))

  return (centroids, context_defects)


def get_contour_list(contours, FAR_ENOUGH_THRESHOLD=10):

  contour_list = []

  for idx, contour in enumerate(contours):
    cnt = contour
    moment = cv.moments(contour)

    if not moment["m00"] == 0.0:
      center_x = moment["m10"] / moment["m00"]
      center_y = moment["m01"] / moment["m00"]
    else:
      x_coords = [x[0][0] for x in contour]
      y_coords = [y[0][1] for y in contour]

      center_x = (np.amin(x_coords) + np.amax(x_coords)) / 2
      center_y = (np.amin(y_coords) + np.amax(y_coords)) / 2

    new_centroid = np.asarray((center_x, center_y))

    contour_list.append([[idx], [new_centroid], cnt])

  curr_idx = 0

  while curr_idx < len(contour_list):
    curr_defect_group = contour_list[curr_idx]

    for curr_centroid in contour_list[curr_idx][1]:
      idx = curr_idx + 1

      while idx < len(contour_list):
        dist = np.sqrt(np.sum(np.square(contour_list[idx][1][0] - curr_centroid)))

        if dist < FAR_ENOUGH_THRESHOLD:
          contour_list[curr_idx][0].append(contour_list[idx][0][0])
          contour_list[curr_idx][1].append(contour_list.pop(idx)[1][0])

        idx += 1

    curr_idx += 1

  return contour_list


def isolate_parts(img: np.ndarray) -> np.ndarray:
  # Remove green channel (code gotten from: https://pythonexamples.org/python-opencv-remove-green-channel-from-color-image/)
  green_image = copy.deepcopy(img)
  green_image[:,:,1] = np.zeros([img.shape[0], img.shape[1]])

  # Threshold/binarize image
  gray_img = cv2.cvtColor(green_image, cv2.COLOR_BGR2GRAY)
  _, thresholded = cv2.threshold(gray_img, 20, 255, cv2.THRESH_BINARY)

  # Eliminate uneeded details (vertical and horizontal lines)
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,7))
  morph_img = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel)

  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,5))
  morph_img = cv2.morphologyEx(morph_img, cv2.MORPH_OPEN, kernel)

  # Repair soldering points (aggresively)
  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
  aggres_repair = 255 - cv2.morphologyEx(255 - morph_img, cv2.MORPH_OPEN, kernel)

  # Normalize threshold and aggressive repair
  thresholded = 1/255 * thresholded
  aggres_repair = 1/255 * aggres_repair

  # Use threshold as a mask to keep good details
  clean_thresh = np.multiply(thresholded, aggres_repair)

  # Final mask
  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
  mask =  cv2.morphologyEx(clean_thresh, cv2.MORPH_CLOSE, kernel)

  isolated_image = copy.deepcopy(img)
  isolated_image[:,:,0] = np.multiply(mask, img[:,:,0])
  isolated_image[:,:,1] = np.multiply(mask, img[:,:,1])
  isolated_image[:,:,2] = np.multiply(mask, img[:,:,2])

  return isolated_image, mask


def extract_soldering_points(img):
  height, width, _ = img.shape
  #img = cv2.resize(img, (int(width/3), int(height/3)))

  isolated_img, binary = isolate_parts(img)

  arr = np.uint8(binary)
  contours, hierarchy = cv2.findContours(arr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

  img_cnt = copy.deepcopy(img)
  cv2.drawContours(img_cnt, contours, -1, (0, 255, 0), 2)
  contour_list = get_contour_list(contours)
  centroids, context_defects = generate_images(contour_list, contours, (0, 0, 255), isolated_img)
  return centroids, context_defects


def create_defect_image(img_path, model):
  img = cv2.imread(img_path, cv2.IMREAD_COLOR)
  centroids, context_defects = extract_soldering_points(img)

  for idx in range(len(context_defects)):
    prediction = model.predict([context_defects[idx].reshape(-1)])[0]
    if prediction == 1:
      start_point = (int(centroids[idx][0]) - 50, int(centroids[idx][1]) - 50)
      end_point = (int(centroids[idx][0]) + 50, int(centroids[idx][1]) + 50)
      color = (0, 0, 255)
      img = cv2.rectangle(img, start_point, end_point, color, 3)

  return img
