### OVERALL FUNCTION
import numpy as np
import cv2 as cv
from keras import models
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def generate_prediction_image(template, defective, model, \
                              padding_width=50, \
                              padding_height=50, \
                              far_enough_threshold=25, \
                              image_width=100, \
                              image_height=100):
    PADDING_WIDTH = padding_width
    PADDING_HEIGHT = padding_height
    FAR_ENOUGH_THRESHOLD = far_enough_threshold
    IMAGE_WIDTH = image_width
    IMAGE_HEIGHT = image_height

    print("Getting countours")
    contours1, contours2 = find_contours(template, defective,
                                         PADDING_WIDTH=padding_width, PADDING_HEIGHT=padding_height)

    print("Getting defect list")
    defect_list1 = get_defect_list(contours1, FAR_ENOUGH_THRESHOLD=FAR_ENOUGH_THRESHOLD)
    defect_list2 = get_defect_list(contours2, FAR_ENOUGH_THRESHOLD=FAR_ENOUGH_THRESHOLD)

    print("Making padded template")
    padded_template = cv.copyMakeBorder(cv.imread(template), PADDING_HEIGHT, PADDING_HEIGHT, PADDING_WIDTH,
                                        PADDING_WIDTH, cv.BORDER_CONSTANT, value=(255, 255, 255))

    print("Generating Images")
    _, context_defects1 = generate_images(defect_list1, contours1, (0, 0, 255), padded_template.copy())
    _, context_defects2 = generate_images(defect_list2, contours2, (0, 0, 255), padded_template.copy())

    print("Getting predictions")
    predictions1 = predict(context_defects1, model)
    predictions2 = predict(context_defects2, model)

    box_predictions = []

    print("Going to prediction loop1!")
    for idx, prediction in enumerate(predictions1):
        center = defect_list1[idx][1][0]

        left = int(center[0] - image_width / 2)
        top = int(center[1] - image_height / 2)
        right = int(center[0] + image_width / 2)
        bottom = int(center[1] + image_height / 2)

        box_predictions.append((((left, top), (right, bottom)), prediction))

    print("Getting more predictions!")
    for idx, prediction in enumerate(predictions2):
        center = defect_list2[idx][1][0]

        left = int(center[0] - image_width / 2)
        top = int(center[1] - image_height / 2)
        right = int(center[0] + image_width / 2)
        bottom = int(center[1] + image_height / 2)

        box_predictions.append((((left, top), (right, bottom)), prediction))

    print("Making borders")
    defective_image = cv.copyMakeBorder(cv.imread(defective), PADDING_HEIGHT, PADDING_HEIGHT, PADDING_WIDTH,
                                        PADDING_WIDTH, cv.BORDER_CONSTANT, value=(255, 255, 255))

    print("Drawing contors")
    cv.drawContours(defective_image, contours1, -1, (0, 0, 255), -1)
    cv.drawContours(defective_image, contours2, -1, (0, 0, 255), -1)

    print("Adding stuff to image")
    for val in box_predictions:
        text_position = (val[0][0][0] + 5, val[0][0][1] + 15)
        cv.rectangle(defective_image, val[0][0], val[0][1], (0, 0, 255), 2)

        cv.putText(defective_image, val[1], text_position, cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))

    print("Returning values")
    return defective_image


### NEURAL NETWORK

def load_model(filename):
    return models.load_model(filename)


def predict(images, model):
    labels = ["open", "short", "mousebite", "spur", "copper", "pin-hole"]
    predictions = []
    print(len(images))
    for image in images:
        print("Trying image")
        predictions.append(labels[np.argmax(model.predict(np.reshape(image, (1, 100, 100, 3))), axis=1)[0]])

    return predictions


### IMAGE PROCESSING

def find_contours(temp_image_path, test_image_path, PADDING_WIDTH=50, PADDING_HEIGHT=50):
    im = cv.imread(temp_image_path)
    im2 = cv.imread(test_image_path)

    sub1 = cv.subtract(im, im2)
    sub1_expanded = cv.copyMakeBorder(sub1, PADDING_HEIGHT, PADDING_HEIGHT, PADDING_WIDTH, PADDING_WIDTH,
                                      cv.BORDER_CONSTANT, value=0)

    sub2 = cv.subtract(im2, im)
    sub2_expanded = cv.copyMakeBorder(sub2, PADDING_HEIGHT, PADDING_HEIGHT, PADDING_WIDTH, PADDING_WIDTH,
                                      cv.BORDER_CONSTANT, value=0)

    sub1_filter_pre = cv.morphologyEx(sub1_expanded, cv.MORPH_OPEN,
                                      np.asarray([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]], np.uint8))
    sub1_filter = cv.morphologyEx(sub1_filter_pre, cv.MORPH_OPEN,
                                  np.asarray([[1., 0., 1.], [0., 1., 0.], [1., 0., 1.]], np.uint8))

    sub2_filter_pre = cv.morphologyEx(sub2_expanded, cv.MORPH_OPEN,
                                      np.asarray([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]], np.uint8))
    sub2_filter = cv.morphologyEx(sub2_filter_pre, cv.MORPH_OPEN,
                                  np.asarray([[1., 0., 1.], [0., 1., 0.], [1., 0., 1.]], np.uint8))

    _, sub1_thresh = cv.threshold(sub1_filter, 127, 255, cv.THRESH_BINARY)
    _, sub2_thresh = cv.threshold(sub2_filter, 127, 255, cv.THRESH_BINARY)

    imgray1 = cv.cvtColor(sub1_thresh, cv.COLOR_BGR2GRAY)
    imgray2 = cv.cvtColor(sub2_thresh, cv.COLOR_BGR2GRAY)

    ret1, thresh1 = cv.threshold(imgray1, 127, 255, 0)
    ret2, thresh2 = cv.threshold(imgray2, 127, 255, 0)

    contours1, hierarchy1 = cv.findContours(imgray1, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    contours2, hierarchy2 = cv.findContours(imgray2, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    return (contours1, contours2)


def get_defect_list(contours, FAR_ENOUGH_THRESHOLD=10):
    defect_list = []

    for idx, contour in enumerate(contours):
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

        defect_list.append([[idx], [new_centroid]])

    curr_idx = 0

    while curr_idx < len(defect_list):
        for curr_centroid in defect_list[curr_idx][1]:
            idx = curr_idx + 1

            while idx < len(defect_list):
                dist = np.sqrt(np.sum(np.square(defect_list[idx][1][0] - curr_centroid)))

                if dist < FAR_ENOUGH_THRESHOLD:
                    defect_list[curr_idx][0].append(defect_list[idx][0][0])
                    defect_list[curr_idx][1].append(defect_list.pop(idx)[1][0])

                idx += 1

        curr_idx += 1

    return defect_list


def generate_images(defect_list, contours, color, template_image, IMAGE_WIDTH=100, IMAGE_HEIGHT=100):
    ml_defects = []
    context_defects = []

    outside_image_width = len(template_image[0])
    outside_image_height = len(template_image[1])

    black_bg = np.zeros((outside_image_height, outside_image_width, 3))

    for defect in defect_list:
        ml_image = black_bg.copy()
        context_image = template_image.copy()

        total_centroid = np.asarray((0.0, 0.0))

        for centroid in defect[1]:
            total_centroid += centroid

        for idx in defect[0]:
            cv.drawContours(ml_image, contours, idx, color, -1)
            cv.drawContours(context_image, contours, idx, color, -1)

        total_centroid /= len(defect[1])

        l_side_x = int(total_centroid[0] - IMAGE_WIDTH / 2)
        r_side_x = int(total_centroid[0] + IMAGE_WIDTH / 2)
        top_side_y = int(total_centroid[1] - IMAGE_HEIGHT / 2)
        bottom_side_y = int(total_centroid[1] + IMAGE_HEIGHT / 2)

        ml_defects.append(ml_image[top_side_y:bottom_side_y, l_side_x:r_side_x])
        context_defects.append(context_image[top_side_y:bottom_side_y, l_side_x:r_side_x])

    return (ml_defects, context_defects)
