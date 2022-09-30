# The brainExtraction.py reads all the images (images those end with word “thresh”) from the given data and perform the brain slice extraction and brain boundary extraction.

import cv2
import glob
import os

thresh_images = []
filecount = 0

filepath = r"C:\Users\abawane\PycharmProjects\BrainBoundaryExtraction\Data"

for file_name in os.listdir(filepath):
    if file_name.endswith('thresh.png'):
        img = cv2.imread(os.path.join(filepath, file_name))
        if img is not None:
            thresh_images.append(img)
        print(file_name)
        filecount += 1

print("Number of thresh files : ", filecount)
print(len(thresh_images))
