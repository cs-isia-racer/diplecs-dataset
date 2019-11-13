import cv2
import csv
import math
import os
import matplotlib.pyplot as plt
import numpy as np


def process_image(img):

    th = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 51, -30
    )

    return th


# Extract labels from csv
raw_labels = csv.reader(open("224_dataset/control/1315584123.dat"), delimiter="\t")
d = {}

for row in raw_labels:
    d[row[0]] = (row[1], row[2])
raw_labels = [x[1] for x in sorted(d.items(), key=lambda x: x[0])]

# Process labels
mini = 12799.0
maxi = 53052.0

labels = []

for power, angle in raw_labels:
    angle = float(angle)
    angle = ((angle - mini) / (maxi - mini)) * 2 - 1
    labels.append((power, angle))

# Process images
image_files = sorted(os.listdir("224_dataset/images"))[:1000]

fourcc = cv2.VideoWriter_fourcc(*"MJPG")
out = cv2.VideoWriter("output.avi", fourcc, 30.0, (224, 168), 0)

for i in range(len(image_files)):
    image_file = image_files[i]
    label = labels[i]

    image = cv2.imread("224_dataset/images/" + image_file, 0)
    result = process_image(image)

    cv2.putText(
        result,
        "%.2f" % label[1],
        (20, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (128, 128, 128),
        2,
        cv2.LINE_AA,
    )

    cv2.arrowedLine(
        result,
        (112, 167),
        (int(112 + 50 * math.sin(-label[1])), int(167 - 50 * math.cos(-label[1]))),
        (128, 128, 128),
        2,
    )

    out.write(result)

out.release()
