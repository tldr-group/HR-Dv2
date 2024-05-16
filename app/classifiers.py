import numpy as np
from PIL import Image
from multiprocessing import Queue

import torch
from torch.nn.functional import interpolate
import hr_dv2.transform as tr
from hr_dv2.utils import *
from hr_dv2 import HighResDV2

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from features import DEAFAULT_WEKA_FEATURES, multiscale_advanced_features


def flatten_mask_training_data(
    feature_stack: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Given $feature_stack and $labels, flatten both and reshape accordingly. Add a class offset if using XGB gpu."""
    h, w, feat = feature_stack.shape
    flat_labels = labels.reshape((h * w))
    flat_features = feature_stack.reshape((h * w, feat))
    labelled_mask = np.nonzero(flat_labels)

    fit_data = flat_features[labelled_mask[0], :]
    target_data = flat_labels[labelled_mask[0]]
    return fit_data, target_data


class Model:
    def __init__(self, send_queue: Queue, recv_queue: Queue) -> None:
        self.send_queue = send_queue
        self.recv_queue = recv_queue

        self.classifier: LogisticRegression | RandomForestClassifier = (
            LogisticRegression("l2", n_jobs=12, max_iter=1000, warm_start=True)
        )

    def get_features(
        self, images: list[Image.Image], inds: list[int], send: bool = True
    ) -> list[np.ndarray]:
        features: list[np.ndarray] = []
        for img, i in zip(images, inds):
            feats = self.img_to_features(img)
            if send:
                self.send_queue.put({f"features_{i}": [feats]})
            features.append(feats)
        return features

    def img_to_features(self, img: Image.Image) -> np.ndarray:
        # overwrite this
        feats = np.zeros((img.height, img.width, 384), dtype=np.float16)
        return feats

    def get_training_data(
        self, features: list[np.ndarray], labels: list[np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        init = False
        for label, feat in zip(labels, features):
            if init == False:
                fit_data, target_data = flatten_mask_training_data(feat, label)
                all_fit_data = fit_data
                all_target_data = target_data
                init = True
            else:
                fit_data, target_data = flatten_mask_training_data(feat, label)
                all_fit_data = np.concatenate((all_fit_data, fit_data), axis=0)
                all_target_data = np.concatenate((all_target_data, target_data), axis=0)
        return all_fit_data, all_target_data

    def train(
        self, features: list[np.ndarray], labels: list[np.ndarray], send: bool = True
    ) -> None:
        fit_data, target_data = self.get_training_data(features, labels)
        self.classifier.fit(fit_data, target_data)
        if send:
            self.send_queue.put({"train_complete": "_"})

    def segment(
        self, features: list[np.ndarray], inds: list[int], send: bool = True
    ) -> list[np.ndarray]:
        segmentations: list[np.ndarray] = []
        for feat, i in zip(features, inds):
            h, w, c = feat.shape
            flat_features = feat.reshape((h * w, c))
            flat_probs = self.classifier.predict_proba(flat_features)
            flat_preds = np.argmax(flat_probs, axis=-1).astype(np.uint8) + 1
            seg = flat_preds.reshape((h, w))
            if send:
                self.send_queue.put({f"segmentation_{i}": [seg]})
            segmentations.append(seg)

        return segmentations


class DeepFeaturesModel(Model):
    def __init__(self, send_queue: Queue, recv_queue: Queue, model_name: str) -> None:
        super().__init__(send_queue, recv_queue)

        self.net = HighResDV2(model_name, 4, pca_dim=-1, dtype=16)
        self.net.cuda()
        self.net.eval()

        shift_dists = [i for i in range(1, 2)]
        fwd_shift, inv_shift = tr.get_shift_transforms(shift_dists, "Moore")
        self.net.set_transforms(fwd_shift, inv_shift)

    def img_to_features(self, img: Image.Image) -> np.ndarray:
        rgb_pil_img = img.convert("RGB")
        tensor: torch.Tensor = tr.to_norm_tensor(rgb_pil_img)
        tensor = tensor.cuda()
        feats = self.net.forward_sequential(tensor)
        feats = interpolate(feats, (rgb_pil_img.height, rgb_pil_img.width))
        feats_np = tr.to_numpy(feats)
        return feats_np.transpose((1, 2, 0))


class WekaFeaturesModel(Model):
    def __init__(self, send_queue: Queue, recv_queue: Queue) -> None:
        super().__init__(send_queue, recv_queue)
        self.classifier = RandomForestClassifier(
            n_estimators=200, max_features=2, max_depth=10, n_jobs=12, warm_start=True
        )

    def img_to_features(self, img: Image.Image) -> np.ndarray:
        greyscale = img.convert("L")
        arr = np.array(greyscale)
        feats = multiscale_advanced_features(arr, DEAFAULT_WEKA_FEATURES)
        return feats


def get_featuriser_classifier(name: str, send_queue: Queue, recv_queue: Queue) -> Model:
    if name == "DINOv2-S-14":
        return DeepFeaturesModel(send_queue, recv_queue, "dinov2_vits14_reg")
    elif name == "DINO-S-8":
        return DeepFeaturesModel(send_queue, recv_queue, "dino_vits8")
    else:
        return WekaFeaturesModel(send_queue, recv_queue)
