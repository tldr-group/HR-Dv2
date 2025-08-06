import numpy as np
from PIL import Image
from multiprocessing import Queue
import pickle
from skops.io import load as skload

import torch
from torch.nn.functional import interpolate
import hr_dv2.transform as tr
from hr_dv2.utils import *
from hr_dv2 import HighResDV2
from hr_dv2.segment import default_crf_params, _get_crf
from pydensecrf.utils import unary_from_softmax

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from hr_dv2.wss.features import DEAFAULT_WEKA_FEATURES, multiscale_advanced_features

from typing import Literal

Transforms = Literal["shift", "flip", "both", None]
AllowedFeaturisers = Literal["DINOv2-S-14", "DINO-S-8", "FeatUp", "hybrid", "hybrid_featup", "bilinear", "weka"]


def flatten_mask_training_data(feature_stack: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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

        self.classifier: LogisticRegression | RandomForestClassifier = LogisticRegression(
            "l2",
            n_jobs=12,
            max_iter=1000,
            warm_start=False,
            class_weight="balanced",
        )
        self.do_crf: bool = True

    def get_features(self, images: list[Image.Image], inds: list[int], send: bool = True) -> list[np.ndarray]:
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

    def get_training_data(self, features: list[np.ndarray], labels: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
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

    def train(self, features: list[np.ndarray], labels: list[np.ndarray], send: bool = True) -> None:
        fit_data, target_data = self.get_training_data(features, labels)
        self.classifier.fit(fit_data, target_data)
        if send:
            self.send_queue.put({"train_complete": "_"})

    def segment(
        self,
        features: list[np.ndarray],
        imgs: list[Image.Image],
        inds: list[int],
        send: bool = True,
    ) -> list[np.ndarray]:
        segmentations: list[np.ndarray] = []
        for feat, i in zip(features, inds):
            h, w, c = feat.shape
            flat_features = feat.reshape((h * w, c))
            flat_probs = self.classifier.predict_proba(flat_features)

            seg: np.ndarray
            if self.do_crf:
                seg = self.crf(imgs[i], flat_probs)
            else:
                flat_preds = np.argmax(flat_probs, axis=-1).astype(np.uint8) + 1
                seg = flat_preds.reshape((h, w))
            if send:
                self.send_queue.put({f"segmentation_{i}": [seg]})
            segmentations.append(seg)
        return segmentations

    def crf(self, pil_img: Image.Image, probs: np.ndarray) -> np.ndarray:
        img_arr = np.array(pil_img)
        h, w = pil_img.height, pil_img.width
        unary = unary_from_softmax(probs).T
        d = _get_crf(img_arr, probs.shape[-1], unary, default_crf_params)
        Q = d.inference(default_crf_params.n_infer)
        crf_seg = np.argmax(Q, axis=0) + 1
        crf_seg = crf_seg.reshape((h, w))
        return crf_seg

    # I/O
    def save_model(self, file_obj) -> None:
        pickle.dump(self.classifier, file_obj)

    def load_model(self, path: str) -> None:
        if ".pkl" in path.lower():
            with open(path, "rb") as f:
                self.classifier = pickle.load(f)
        else:
            self.classifier = skload(path)


class DeepFeaturesModel(Model):
    def __init__(
        self,
        send_queue: Queue,
        recv_queue: Queue,
        model_name: str,
        trs: Transforms = None,
        bilinear: bool = False,
    ) -> None:
        super().__init__(send_queue, recv_queue)

        self.net = HighResDV2(model_name, 4, pca_dim=-1, dtype=16)
        self.net.cuda()
        self.net.eval()

        shift_dists = [i for i in range(1, 3)]
        fwd_shift, inv_shift = tr.get_shift_transforms(shift_dists, "Moore")
        fwd_flip, inv_flip = tr.get_flip_transforms()

        if trs == "both":
            fwd, inv = tr.combine_transforms(fwd_shift, fwd_flip, inv_shift, inv_flip)
        elif trs == "shift":
            fwd, inv = fwd_shift, inv_shift
        elif trs == "flip":
            fwd, inv = fwd_flip, inv_flip
        else:
            fwd, inv = [], []

        self.net.set_transforms(fwd, inv)

        if bilinear:
            self.net = HighResDV2(model_name, 14, pca_dim=-1, dtype=16)
            self.net.interpolation_mode = "bilinear"
            self.net.cuda()
            self.net.eval()
            self.net.set_transforms([], [])

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
            n_estimators=200,
            max_features=2,
            max_depth=10,
            n_jobs=12,
            warm_start=False,
            class_weight="balanced",
        )

    def img_to_features(self, img: Image.Image) -> np.ndarray:
        greyscale = img.convert("L")
        arr = np.array(greyscale)
        feats = multiscale_advanced_features(arr, DEAFAULT_WEKA_FEATURES)
        return feats


class FeatUp(DeepFeaturesModel):
    def __init__(self, send_queue: Queue, recv_queue: Queue, model_name: str) -> None:
        super().__init__(send_queue, recv_queue, model_name)

        self.net = torch.hub.load("mhamilton723/FeatUp", "dinov2", use_norm=False)
        self.net.cuda()
        self.net.eval()

    def img_to_features(self, img: Image.Image) -> np.ndarray:
        rgb_pil_img = img.convert("RGB")
        tensor: torch.Tensor = tr.to_norm_tensor(rgb_pil_img)
        tensor = tensor.cuda()
        feats = self.net.forward(tensor.unsqueeze(0))
        feats = interpolate(feats, (rgb_pil_img.height, rgb_pil_img.width))
        feats_np = tr.to_numpy(feats)

        return feats_np.transpose((1, 2, 0))


class Hybrid(DeepFeaturesModel):
    def __init__(
        self,
        send_queue: Queue,
        recv_queue: Queue,
        model_name: str,
        trs: Transforms = None,
        bilinear: bool = False,
    ) -> None:
        super().__init__(send_queue, recv_queue, model_name, trs)
        """
        self.classifier = RandomForestClassifier(
            n_estimators=200,
            max_features=2,
            max_depth=10,
            n_jobs=12,
            warm_start=False,
            class_weight="balanced",
        )
        """

    def img_to_features(self, img: Image.Image) -> np.ndarray:
        rgb_pil_img = img.convert("RGB")
        tensor: torch.Tensor = tr.to_norm_tensor(rgb_pil_img)
        tensor = tensor.cuda()
        feats = self.net.forward_sequential(tensor)
        feats = interpolate(feats, (rgb_pil_img.height, rgb_pil_img.width))
        deep_feats = tr.to_numpy(feats).transpose((1, 2, 0))

        greyscale = img.convert("L")
        arr = np.array(greyscale)
        classical_feats = multiscale_advanced_features(arr, DEAFAULT_WEKA_FEATURES)
        hybrid_feats = np.concatenate((deep_feats, classical_feats), axis=-1)
        # print(hybrid_feats.shape)
        # print(self.classifier)
        hybrid_feats = rescale_pca_img(hybrid_feats)
        return hybrid_feats


class HybridFeatUp(DeepFeaturesModel):
    def __init__(self, send_queue: Queue, recv_queue: Queue, model_name: str) -> None:
        super().__init__(send_queue, recv_queue, model_name)

        self.net = torch.hub.load("mhamilton723/FeatUp", "dinov2", use_norm=False)
        self.net.cuda()
        self.net.eval()

    def img_to_features(self, img: Image.Image) -> np.ndarray:
        rgb_pil_img = img.convert("RGB")
        tensor: torch.Tensor = tr.to_norm_tensor(rgb_pil_img)
        tensor = tensor.cuda()
        feats = self.net.forward(tensor.unsqueeze(0))
        feats = interpolate(feats, (rgb_pil_img.height, rgb_pil_img.width))
        deep_feats = tr.to_numpy(feats).transpose((1, 2, 0))

        greyscale = img.convert("L")
        arr = np.array(greyscale)
        classical_feats = multiscale_advanced_features(arr, DEAFAULT_WEKA_FEATURES)
        hybrid_feats = np.concatenate((deep_feats, classical_feats), axis=-1)
        # print(hybrid_feats.shape)
        # print(self.classifier)
        hybrid_feats = rescale_pca_img(hybrid_feats)

        return hybrid_feats


def get_featuriser_classifier(
    name: AllowedFeaturisers,
    send_queue: Queue,
    recv_queue: Queue,
    trs: Transforms = None,
) -> Model:
    if name == "DINOv2-S-14":
        return DeepFeaturesModel(send_queue, recv_queue, "dinov2_vits14_reg", trs)
    elif name == "DINO-S-8":
        return DeepFeaturesModel(send_queue, recv_queue, "dino_vits8", trs)
    elif name == "FeatUp":
        return FeatUp(send_queue, recv_queue, "dinov2_vits14_reg")
    elif name == "hybrid":
        return Hybrid(send_queue, recv_queue, "dinov2_vits14_reg", trs)
    elif name == "hybrid_featup":
        return HybridFeatUp(send_queue, recv_queue, "dinov2_vits14_reg")
    elif name == "bilinear":
        return DeepFeaturesModel(send_queue, recv_queue, "dinov2_vits14_reg", trs, bilinear=True)
    else:
        return WekaFeaturesModel(send_queue, recv_queue)
