from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os
import os.path as osp
import glob
import pickle

from ..dataset import ImageDataset


class CUSTOM_Dataset(ImageDataset):
    dataset_dir = 'cdataset'

    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        # All you need to do here is to generate three lists,
        # which are train, query and gallery.
        # Each list contains tuples of (img_path, pid, camid),
        # where
        # - img_path (str): absolute path to an image.
        # - pid (int): person ID, e.g. 0, 1.
        # - camid (int): camera ID, e.g. 0, 1.
        # Note that
        # - pid and camid should be 0-based.
        # - query and gallery should share the same pid scope (e.g.
        #   pid=0 in query refers to the same person as pid=0 in gallery).
        # - train, query and gallery share the same camid scope (e.g.
        #   camid=0 in train refers to the same camera as camid=0
        #   in query/gallery).
        
        train_files = glob.glob(os.path.join(self.dataset_dir, 'train/*.jpg'))
        query_files = glob.glob(os.path.join(self.dataset_dir, 'query/*.jpg'))
        gallery_files = glob.glob(os.path.join(self.dataset_dir, 'gallery/*.jpg'))
        train, query, gallery = [], [], []

        with open(os.path.join(self.dataset_dir, 'train_idx_con.pickle'), "rb") as file:
            train_idx_con = pickle.load(file)

        with open(os.path.join(self.dataset_dir, 'test_idx_con.pickle'), "rb") as file:
            test_idx_con = pickle.load(file)

        ## In out dataset, id starts at index 9. So we shoud minus 9

        for train_file in train_files:
            file_name = os.path.basename(train_file).split('_')
            if file_name[1] == 'cctv02':
                camera = 0
            elif file_name[1] == 'cctv03':
                camera = 1
            img_id = train_idx_con[int(file_name[3])]

            train.append((train_file, img_id, camera))

        for query_file in query_files:
            file_name = os.path.basename(query_file).split('_')
            if file_name[1] == 'cctv02':
                camera = 0
            elif file_name[1] == 'cctv03':
                camera = 1
            img_id = test_idx_con[int(file_name[3])]

            query.append((query_file, img_id, camera))

        for gallery_file in gallery_files:
            file_name = os.path.basename(gallery_file).split('_')
            if file_name[1] == 'cctv02':
                camera = 0
            elif file_name[1] == 'cctv03':
                camera = 1
            img_id = test_idx_con[int(file_name[3])]

            gallery.append((gallery_file, img_id, camera))


        super(CUSTOM_Dataset, self).__init__(train, query, gallery, **kwargs)
