import os
import glob
from PIL import Image

label_files = glob.glob('reid-data/cdataset/label/*.txt') # hard coded path

os.makedirs('reid-data/cdataset/train') # hard coded path
os.makedirs('reid-data/cdataset/test') # hard coded path

# hard coded path
with open('reid-data/cdataset/train_ids.txt', 'r') as tr:
    train_ids = tr.readline().split(',')[:-1]

with open('reid-data/cdataset/test_ids.txt', 'r') as te:
    test_ids = te.readline().split(',')[:-1]

for label_file in label_files:
    
    print(label_file[:-4])
    with open(label_file, 'r') as f:
        labels = f.readlines()

    for label in labels:
        s_label = label.split(', ')
        frame = s_label[0]
        img_id = s_label[1]
        left, top, width, height = float(s_label[2]), float(s_label[3]), float(s_label[4]), float(s_label[5])

        img_path = label_file[:-4].replace('label', 'image') + '_' + frame + '.jpg'
        img = Image.open(img_path)
        box = (left, top, left+width, top+height)

        crop_img = img.crop(box)
        if img_id in train_ids:
            save_path = label_file[:-4].replace('label', 'train') + '_' + img_id + '_' + frame + '.jpg'

        if img_id in test_ids:
            save_path = label_file[:-4].replace('label', 'test') + '_' + img_id + '_' + frame + '.jpg'

        crop_img.save(save_path)









