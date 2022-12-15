import os
import glob
import shutil
import random
from sklearn.model_selection import train_test_split


test_image_paths = glob.glob('reid-data/cdataset/test/*.jpg')
image_per_id = dict()

os.makedirs('reid-data/cdataset/gallery')
os.makedirs('reid-data/cdataset/query')

for test_image_path in test_image_paths:
    file_name = os.path.basename(test_image_path)
    file_name = file_name.split('_')
    camera = file_name[1]
    img_id = file_name[3]

    if img_id not in image_per_id.keys():
        image_per_id[img_id] = {'cctv02': [], 'cctv03': []}

    image_per_id[img_id][camera].append(test_image_path)

q = open('reid-data/cdataset/query.txt', 'w')
g = open('reid-data/cdataset/gallery.txt', 'w')


test_id_list = list(image_per_id.keys())
for test_id in test_id_list:
    images_in_c2 = image_per_id[test_id]['cctv02']

    if len(images_in_c2) > 2:
        query_path = random.choice(images_in_c2)
        q.write(os.path.basename(query_path)+'\n')

        for img_in_c2 in images_in_c2:
            if img_in_c2 == query_path:
                shutil.copy(img_in_c2, img_in_c2.replace('test', 'query'))
            else:
                shutil.copy(img_in_c2, img_in_c2.replace('test', 'gallery'))
                g.write(os.path.basename(img_in_c2)+'\n')

    images_in_c3 = image_per_id[test_id]['cctv03']

    if len(images_in_c3) > 2:
        query_path = random.choice(images_in_c3)
        q.write(os.path.basename(query_path)+'\n')

        for img_in_c3 in images_in_c3:
            if img_in_c3 == query_path:
                shutil.copy(img_in_c3, img_in_c3.replace('test', 'query'))
            else:
                shutil.copy(img_in_c3, img_in_c3.replace('test', 'gallery'))
                g.write(os.path.basename(img_in_c3)+'\n')

q.close()
g.close()












