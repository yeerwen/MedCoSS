import os
import cv2
from tqdm import tqdm
import random
root_path = "/media/new_userdisk0/Continual_pretraining/2D/QaTa_CoV19/QaTa-COV19-v2/"

Train_Set_path = os.path.join(root_path, "Train Set")
Train_img_path = os.path.join(Train_Set_path, "Images")
Train_lab_path = os.path.join(Train_Set_path, "Ground-truths")

Test_Set_path = os.path.join(root_path, "Test Set")
Test_img_path = os.path.join(Test_Set_path, "Images")
Test_lab_path = os.path.join(Test_Set_path, "Ground-truths")

shape_set = set()

train_txt = open(os.path.join(root_path, "train.txt"), "w")
val_txt = open(os.path.join(root_path, "val.txt"), "w")
test_txt = open(os.path.join(root_path, "test.txt"), "w")
train_path = os.listdir(Train_img_path)
random.seed(0)
random.shuffle(train_path)
for name in tqdm(train_path[:int(0.8*len(train_path))]):
    img = cv2.imread(os.path.join(Train_img_path, name), flags=0)
    lab = cv2.imread(os.path.join(Train_lab_path, "mask_"+name), flags=0)
    assert img.shape == lab.shape
    shape_set.add(img.shape)
    train_txt.write(name+"\n")

for name in tqdm(train_path[int(0.8*len(train_path)):]):
    img = cv2.imread(os.path.join(Train_img_path, name), flags=0)
    lab = cv2.imread(os.path.join(Train_lab_path, "mask_"+name), flags=0)
    assert img.shape == lab.shape
    shape_set.add(img.shape)
    val_txt.write(name+"\n")

for name in tqdm(os.listdir(Test_img_path)):
    img = cv2.imread(os.path.join(Test_img_path, name), flags=0)
    lab = cv2.imread(os.path.join(Test_lab_path, "mask_"+name), flags=0)
    assert img.shape == lab.shape
    shape_set.add(img.shape)
    test_txt.write(name+"\n")

train_txt.close()
val_txt.close()
test_txt.close()
print(shape_set)


