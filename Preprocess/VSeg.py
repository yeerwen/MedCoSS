import os
import json

root_path = "/media/new_userdisk0/Continual_pretraining/3D/Task061_VSseg/"

train_list = open(os.path.join(root_path, "VSseg_train.txt"), "w")
test_list = open(os.path.join(root_path, "VSseg_test.txt"), "w")
val_index = 0
all_patient = []
with open(os.path.join(root_path, "dataset.json"), "r", encoding="utf-8") as f:
    content = json.load(f)
    print(content["training"])
    for sub_content in content["training"]:
        image_name = sub_content["image"].split("/")[-1].split(".nii")[0]
        all_patient.append(image_name)

print(all_patient, len(all_patient))
import random
random.seed(0)
random.shuffle(all_patient)
for name in all_patient[:int(0.8*len(all_patient))]:
    train_list.write(name+"\n")
for name in all_patient[int(0.8*len(all_patient)):]:
    test_list.write(name+"\n")
train_list.close()
test_list.close()
