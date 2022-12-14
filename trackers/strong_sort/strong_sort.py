import numpy as np
import torch
import sys
import cv2
import gdown
from os.path import exists as file_exists, join
import torchvision.transforms as transforms

from sort.nn_matching import NearestNeighborDistanceMetric
from sort.detection import Detection
from sort.tracker import Tracker

from reid_multibackend import ReIDDetectMultiBackend

from yolov5.utils.general import xyxy2xywh
from PIL import Image


class StrongSORT(object):
    def __init__(self, 
                 model_weights,
                 device,
                 fp16,
                 max_dist=0.2,
                 max_iou_distance=0.7,
                 max_age=70, n_init=3,
                 nn_budget=100,
                 mc_lambda=0.995,
                 ema_alpha=0.9
                ):

        self.model = ReIDDetectMultiBackend(weights=model_weights, device=device, fp16=fp16)
        self.max_dist = max_dist
        metric = NearestNeighborDistanceMetric(
            "cosine", self.max_dist, nn_budget)
        self.tracker = Tracker(
            metric, max_iou_distance=max_iou_distance, max_age=max_age, n_init=n_init)

    def update(self, dets,  ori_img):
        
        xyxys = dets[:, 0:4]
        confs = dets[:, 4]
        clss = dets[:, 5]

        classes = clss.numpy()
        xywhs = xyxy2xywh(xyxys.numpy())
        confs = confs.numpy()
        self.height, self.width = ori_img.shape[:2]
        
        # generate detections
        features = self._get_features(xywhs, ori_img)
        bbox_tlwh = self._xywh_to_tlwh(xywhs)
        detections = [Detection(bbox_tlwh[i], conf, features[i]) for i, conf in enumerate(
            confs)]

        # run on non-maximum supression
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])

        # update tracker
        self.tracker.predict()
        self.tracker.update(detections, clss, confs)

        # output bbox identities
        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            box = track.to_tlwh()
            x1, y1, x2, y2 = self._tlwh_to_xyxy(box)
            
            track_id = track.track_id
            class_id = track.class_id
            conf = track.conf
            outputs.append(np.array([x1, y1, x2, y2, track_id, class_id, conf]))
        if len(outputs) > 0:
            outputs = np.stack(outputs, axis=0)
        return outputs

    """
    TODO:
        Convert bbox from xc_yc_w_h to xtl_ytl_w_h
    Thanks JieChen91@github.com for reporting this bug!
    """
    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        elif isinstance(bbox_xywh, torch.Tensor):
            bbox_tlwh = bbox_xywh.clone()
        bbox_tlwh[:, 0] = bbox_xywh[:, 0] - bbox_xywh[:, 2] / 2.
        bbox_tlwh[:, 1] = bbox_xywh[:, 1] - bbox_xywh[:, 3] / 2.
        return bbox_tlwh

    def _xywh_to_xyxy(self, bbox_xywh):
        x, y, w, h = bbox_xywh
        x1 = max(int(x - w / 2), 0)
        x2 = min(int(x + w / 2), self.width - 1)
        y1 = max(int(y - h / 2), 0)
        y2 = min(int(y + h / 2), self.height - 1)
        return x1, y1, x2, y2

    def _tlwh_to_xyxy(self, bbox_tlwh):
        """
        TODO:
            Convert bbox from xtl_ytl_w_h to xc_yc_w_h
        Thanks JieChen91@github.com for reporting this bug!
        """
        x, y, w, h = bbox_tlwh
        x1 = max(int(x), 0)
        x2 = min(int(x+w), self.width - 1)
        y1 = max(int(y), 0)
        y2 = min(int(y+h), self.height - 1)
        return x1, y1, x2, y2

    def increment_ages(self):
        self.tracker.increment_ages()

    def _xyxy_to_tlwh(self, bbox_xyxy):
        x1, y1, x2, y2 = bbox_xyxy

        t = x1
        l = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        return t, l, w, h

    def _get_features(self, bbox_xywh, ori_img):
        im_crops = []
        for box in bbox_xywh:
            x1, y1, x2, y2 = self._xywh_to_xyxy(box)
            im = ori_img[y1:y2, x1:x2]
            im_crops.append(im)
        if im_crops:
            features = self.model(im_crops)
        else:
            features = np.array([])
        return features


class StrongSORT_w_Matching(object):
    def __init__(self,
                 model_weights,
                 device,
                 fp16,
                 max_dist=0.2,
                 max_iou_distance=0.7,
                 max_age=70, n_init=3,
                 nn_budget=100,
                 mc_lambda=0.995,
                 ema_alpha=0.9,
                 query_img_path=None
                 ):

        self.model = ReIDDetectMultiBackend(weights=model_weights, device=device, fp16=fp16)
        self.max_dist = max_dist
        metric = NearestNeighborDistanceMetric(
            "cosine", self.max_dist, nn_budget)
        self.tracker = Tracker(
            metric, max_iou_distance=max_iou_distance, max_age=max_age, n_init=n_init)

        self.flag = False  # whether self-driving car is determined
        query_img = [np.asarray(Image.open(query_img_path))]
        self.matching_thres = 0.28
        query_feature = self.model(query_img).detach().cpu().numpy()
        self.query_feature = query_feature / np.linalg.norm(query_feature, axis=1, keepdims=True)



    def update(self, dets, ori_img):

        xyxys = dets[:, 0:4]
        confs = dets[:, 4]
        clss = dets[:, 5]

        classes = clss.numpy()
        xywhs = xyxy2xywh(xyxys.numpy())
        confs = confs.numpy()
        self.height, self.width = ori_img.shape[:2]

        # generate detections
        features = self._get_features(xywhs, ori_img)
        bbox_tlwh = self._xywh_to_tlwh(xywhs)
        detections = [Detection(bbox_tlwh[i], conf, features[i]) for i, conf in enumerate(
            confs)]

        # run on non-maximum supression
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])

        # update tracker
        self.tracker.predict()
        self.tracker.update(detections, clss, confs)

        # classify self-driving car
        if not self.flag:
            img_crops = []
            track_candidate = []
            for track in self.tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue

                box = track.to_tlwh()
                x1, y1, x2, y2 = self._tlwh_to_xyxy(box)
                cen_x = (x1 + x2) / 2
                cen_y = (y1 + y2) / 2

                if (500 < cen_y) and (cen_y < 750):
                    if (372 < cen_x) and (cen_x < 1688):
                        im = ori_img[y1:y2, x1:x2]
                        img_crops.append(im)
                        track_candidate.append(track)

            if img_crops:
                features = self.model(img_crops).detach().cpu().numpy()
                features = features / np.linalg.norm(features, axis=1, keepdims=True)
                dist = 1. - np.dot(self.query_feature, features.T)

                for i in range(np.shape(dist)[1]):
                    if dist[0, i] < self.matching_thres:
                        print('************************************************')
                        print('************************************************')
                        print('************************************************')
                        print('success to classify self-driving car')
                        print('************************************************')
                        print('************************************************')
                        print('************************************************')

                        track_candidate[i].is_AV = True
                        self.flag = True

        # output bbox identities
        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            box = track.to_tlwh()
            x1, y1, x2, y2 = self._tlwh_to_xyxy(box)

            track_id = track.track_id
            if track.is_AV:
                track_id = 88888

            class_id = track.class_id
            conf = track.conf
            outputs.append(np.array([x1, y1, x2, y2, track_id, class_id, conf]))
        if len(outputs) > 0:
            outputs = np.stack(outputs, axis=0)
        return outputs

    """
    TODO:
        Convert bbox from xc_yc_w_h to xtl_ytl_w_h
    Thanks JieChen91@github.com for reporting this bug!
    """

    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        elif isinstance(bbox_xywh, torch.Tensor):
            bbox_tlwh = bbox_xywh.clone()
        bbox_tlwh[:, 0] = bbox_xywh[:, 0] - bbox_xywh[:, 2] / 2.
        bbox_tlwh[:, 1] = bbox_xywh[:, 1] - bbox_xywh[:, 3] / 2.
        return bbox_tlwh

    def _xywh_to_xyxy(self, bbox_xywh):
        x, y, w, h = bbox_xywh
        x1 = max(int(x - w / 2), 0)
        x2 = min(int(x + w / 2), self.width - 1)
        y1 = max(int(y - h / 2), 0)
        y2 = min(int(y + h / 2), self.height - 1)
        return x1, y1, x2, y2

    def _tlwh_to_xyxy(self, bbox_tlwh):
        """
        TODO:
            Convert bbox from xtl_ytl_w_h to xc_yc_w_h
        Thanks JieChen91@github.com for reporting this bug!
        """
        x, y, w, h = bbox_tlwh
        x1 = max(int(x), 0)
        x2 = min(int(x + w), self.width - 1)
        y1 = max(int(y), 0)
        y2 = min(int(y + h), self.height - 1)
        return x1, y1, x2, y2

    def increment_ages(self):
        self.tracker.increment_ages()

    def _xyxy_to_tlwh(self, bbox_xyxy):
        x1, y1, x2, y2 = bbox_xyxy

        t = x1
        l = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        return t, l, w, h

    def _get_features(self, bbox_xywh, ori_img):
        im_crops = []
        for box in bbox_xywh:
            x1, y1, x2, y2 = self._xywh_to_xyxy(box)
            im = ori_img[y1:y2, x1:x2]
            im_crops.append(im)
        if im_crops:
            features = self.model(im_crops)
        else:
            features = np.array([])
        return features
