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


template_file = r"C:\Users\abawane\PycharmProjects\BrainBoundaryExtraction\Data\R_template.png"
template_img = cv2.imread(template_file)
W, H = template_img.shape[:2]


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


test_file = r"C:\Users\abawane\PycharmProjects\BrainBoundaryExtraction\Data\IC_1_TEST_thresh.png"
test_img = cv2.imread(test_file)

img_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
temp_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)

# Passing the image to matchTemplate method
match = cv2.matchTemplate(image=img_gray, templ=temp_gray, method=cv2.TM_CCOEFF_NORMED)

# Select rectangles with
# confidence greater than threshold
(y_points, x_points) = np.where(match >= thresh)

# initialize our list of bounding boxes
boxes = list()

# store co-ordinates of each bounding box
# we'll create a new list by looping
# through each pair of points
for (x, y) in zip(x_points, y_points):
    # print(x, y, sep=' ')
    boxes.append((x, y, x + 124, y + 124))

# boxes = non_max_suppression(np.array(boxes))

print(len(boxes))

count = 0

for (x1, y1, x2, y2) in boxes:
    # draw the bounding box on the image
    cv2.rectangle(img, (x1, y1), (x2, y2), 255, 4)
    count += 1
    roi = test_img[y1:y2, x1:x2]
    # print(x1, y1, x2, y2, sep = ' ')
    cv2.imwrite(str(count) + '.png', roi)
    # cv2.imshow("Template", template_img)
    # cv2.imshow("After NMS", roi)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()