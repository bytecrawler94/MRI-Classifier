import cv2
import time
import glob
import os
from PIL import Image
import numpy as np
from imutils.object_detection import non_max_suppression

start_time = time.time()
thresh = 0.99
thresh_images = []
filecount = 0

try:
    parent_dir = os.getcwd()
    Data_path = parent_dir + r"\\testPatient"
except:
    pass

folder_boundaries = "Boundaries"
folder_slices = "Slices"
try:
    os.mkdir(os.path.join(parent_dir, folder_boundaries))
    os.mkdir(os.path.join(parent_dir, folder_slices))
except:
    pass

template_file = Data_path + r"\\R_template.png"
template_img = cv2.imread(template_file)
temp_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
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

        test_img = img
        img_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

        match = cv2.matchTemplate(image=img_gray, templ=temp_gray, method=cv2.TM_CCOEFF_NORMED)

        (y_points, x_points) = np.where(match >= thresh)

        boxes = list()
        co_ordinates = list()
        freq_x = {}
        freq_y = {}
        for i in range(len(x_points) - 1):
            if x_points[i] in freq_x:
                freq_x[x_points[i]] += 1
            else:
                freq_x[x_points[i]] = 1

            if y_points[i] in freq_y:
                freq_y[y_points[i]] += 1
            else:
                freq_y[y_points[i]] = 1

        x_src = 0
        y_src = 0
        x_dest = 0
        y_dest = 0

        iter = 0
        for key, value in freq_x.items():
            iter += 1
            if iter == 1:
                x_src = key
            elif iter == 2:
                x_dest = key
                break

        width = x_dest - x_src

        iter = 0
        for key, value in freq_y.items():
            iter += 1
            if iter == 1:
                y_src = key
            elif iter == 2:
                y_dest = key
                break

        height = y_dest - y_src

        for (x, y) in zip(x_points, y_points):
            boxes.append((x+W, y+H, x + height, y + width))
            co_ordinates.append((x, y))

        boxes = non_max_suppression(np.array(boxes))
        count = 0

        slices_dir = parent_dir + r"\Slices\\"
        boundaries_dir = parent_dir + r"\Boundaries\\"
        os.chdir(slices_dir + folder_name)

        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(img, (x1, y1), (x2, y2), 0, 1)

            roi = test_img[y1:y2, x1:x2]
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            roi_cpy = roi_gray.copy()
            im = Image.fromarray(np.uint8(roi_cpy))
            if im.getbbox():
                cv2.imwrite(str(count) + '.png', roi)
                count += 1

        os.chdir(boundaries_dir + folder_name)

        count = 0

        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(img, (x1, y1), (x2, y2), 0, 1)

            roi = test_img[y1:y2, x1:x2]
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            ret, binary_thresh = cv2.threshold(roi_gray, 50, 255, cv2.THRESH_BINARY)

            contours, hierarchy = cv2.findContours(image=binary_thresh, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image=roi, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA)
            roi_cpy = roi_gray.copy()
            im = Image.fromarray(np.uint8(roi_cpy))
            if im.getbbox():
                cv2.imwrite(str(count) + '.png', roi)
                count += 1
            cv2.destroyAllWindows()

# end_time = time.time()
# total_time = end_time - start_time
# print("\n" + str(total_time)+" seconds")
