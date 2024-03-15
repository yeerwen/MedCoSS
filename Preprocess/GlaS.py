import os
import cv2
import numpy as np
from collections import Counter
from PIL import Image
root_path = "/Users/erwen/Downloads/Warwick_QU_Dataset/"
out_path = "/Users/erwen/Downloads/Glas/"
csv_path = os.path.join(root_path, "Grade.csv")
train_class_name = {"benign": [], "malignant": []}
test_class_name = {"benign": [], "malignant": []}
csv_lines = open(csv_path, "r").readlines()
# print(csv_lines)

for line in csv_lines:
    name, grade = line.split(",")[0].strip(), line.split(",")[2].strip()
    if "train" in name:
        train_class_name[grade].append(name)
    elif "test" in name:
        test_class_name[grade].append(name)
print("train", len(train_class_name["benign"]), len(train_class_name["malignant"]))
print("test", len(test_class_name["benign"]), len(test_class_name["malignant"]))

os.makedirs(os.path.join(out_path, "train", "images"))
os.makedirs(os.path.join(out_path, "train", "labels"))
os.makedirs(os.path.join(out_path, "test", "images"))
os.makedirs(os.path.join(out_path, "test", "labels"))
for name in train_class_name["benign"]:
    # print(os.path.join(root_path, name))
    image = np.asarray(Image.open(os.path.join(root_path, name+".bmp")).resize((512, 512), Image.BILINEAR)).copy()
    label = np.asarray(Image.open(os.path.join(root_path, name+"_anno.bmp")).resize((512, 512), Image.NEAREST)).copy()
    label[label>0] = 255
    print(image.size, label.size, np.min(label), np.max(label))
    print(Counter(label.flatten()))
    cv2.imwrite(os.path.join(out_path, "train", "images", name+".png"), image)
    cv2.imwrite(os.path.join(out_path, "train", "labels", name+".png"), label)

for name in train_class_name["malignant"]:
    # print(os.path.join(root_path, name))
    image = np.asarray(Image.open(os.path.join(root_path, name+".bmp")).resize((512, 512), Image.BILINEAR)).copy()
    label = np.asarray(Image.open(os.path.join(root_path, name+"_anno.bmp")).resize((512, 512), Image.NEAREST)).copy()
    label[label>0] = 255
    print(image.size, label.size, np.min(label), np.max(label))
    print(Counter(label.flatten()))
    cv2.imwrite(os.path.join(out_path, "train", "images", name+".png"), image)
    cv2.imwrite(os.path.join(out_path, "train", "labels", name+".png"), label)


for name in test_class_name["benign"]:
    # print(os.path.join(root_path, name))
    image = np.asarray(Image.open(os.path.join(root_path, name+".bmp")).resize((512, 512), Image.BILINEAR)).copy()
    label = np.asarray(Image.open(os.path.join(root_path, name+"_anno.bmp")).resize((512, 512), Image.NEAREST)).copy()
    label[label>0] = 255
    print(image.size, label.size, np.min(label), np.max(label))
    print(Counter(label.flatten()))
    cv2.imwrite(os.path.join(out_path, "test", "images", name+".png"), image)
    cv2.imwrite(os.path.join(out_path, "test", "labels", name+".png"), label)

for name in test_class_name["malignant"]:
    # print(os.path.join(root_path, name))
    image = np.asarray(Image.open(os.path.join(root_path, name+".bmp")).resize((512, 512), Image.BILINEAR)).copy()
    label = np.asarray(Image.open(os.path.join(root_path, name+"_anno.bmp")).resize((512, 512), Image.NEAREST)).copy()
    label[label>0] = 255
    print(image.size, label.size, np.min(label), np.max(label))
    print(Counter(label.flatten()))
    cv2.imwrite(os.path.join(out_path, "test", "images", name+".png"), image)
    cv2.imwrite(os.path.join(out_path, "test", "labels", name+".png"), label)