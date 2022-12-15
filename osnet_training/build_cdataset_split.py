import os
import glob
from PIL import Image
from sklearn.model_selection import train_test_split


label_files = glob.glob('reid-data/cdataset/label/*.txt') # hard coded path
id_list = []

for label_file in label_files:

    with open(label_file, 'r') as f:
        labels = f.readlines()

    for label in labels:
        s_label = label.split(', ')
        frame = s_label[0]
        img_id = s_label[1]
        left, top, width, height = float(s_label[2]), float(s_label[3]), float(s_label[4]), float(s_label[5])

        img_id = int(img_id)
        if img_id not in id_list:
            id_list.append(img_id)

id_list = sorted(id_list)
print(len(id_list))
print(id_list)

train_id, test_id = train_test_split(id_list, test_size=0.3, random_state=123) # test_size can be changed

print('train------------------------------------------:', len(train_id))
print(sorted(train_id))
print('test------------------------------------------:', len(test_id))
print(sorted(test_id))


with open('reid-data/cdataset/train_ids.txt', 'w') as f:
    for t_id in train_id:
        f.write(str(t_id)+',')

with open('reid-data/cdataset/test_ids.txt', 'w') as f:
    for t_id in test_id:
        f.write(str(t_id)+',')











