import cv2
import glob
import os
from PIL import Image
import numpy as np
from imutils.object_detection import non_max_suppression

thresh = 0.99
thresh_images = []
filecount = 0

try:
    parent_dir = r"C:\Users\abawane\PycharmProjects\BrainBoundaryExtraction"
    Data_path = r"C:\Users\abawane\PycharmProjects\BrainBoundaryExtraction\Data"
except:
    pass

folder_boundaries = "Boundaries"
folder_slices = "Slices"
try:
    os.mkdir(os.path.join(parent_dir, folder_boundaries))
    os.mkdir(os.path.join(parent_dir, folder_slices))
except:
    pass

for file_name in os.listdir(Data_path):
    if file_name.endswith('thresh.png'):
        folder_name = file_name[:-4]
        try:
            os.mkdir(os.path.join(parent_dir+r"\Boundaries", folder_name))
            os.mkdir(os.path.join(parent_dir+r"\Slices", folder_name))
        except:
            pass

        img = cv2.imread(os.path.join(Data_path, file_name))
        if img is not None:
            thresh_images.append(img)

