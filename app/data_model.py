"""Bread and butter of the app. Holds all the images and numpy array data for each image we want to segment."""

# %% IMPORTS
import numpy as np
from PIL import Image, ImageDraw
from skimage.measure import find_contours, approximate_polygon
from multiprocessing import Process, Queue, set_start_method
from sklearn.linear_model import LogisticRegression

from os import getcwd
from time import sleep


from tifffile import imread, imwrite

from dataclasses import dataclass
from typing import TypeAlias, List, Tuple, Literal, cast

import torch
from torch.nn.functional import interpolate
import hr_dv2.transform as tr
from hr_dv2.utils import *
from hr_dv2 import HighResDV2

set_start_method("spawn", force=True)

# %% TYPES
Point: TypeAlias = Tuple[float | int, float | int]
Polygon: TypeAlias = List[Point]
Rectangle: TypeAlias = Tuple[float | int, float | int, float | int, float | int]
LabelType: TypeAlias = Literal["Polygon", "Brush", "Eraser"]

# %% CONSTANTS
PAD: int = 3
DEAFAULT_FEATURES = {
    "Gaussian Blur": 1,
    "Sobel Filter": 1,
    "Hessian": 1,
    "Difference of Gaussians": 1,
    "Membrane Projections": 1,
    "Mean": 0,
    "Minimum": 0,
    "Maximum": 0,
    "Median": 0,
    "Bilateral": 0,
    "Derivatives": 0,
    "Structure": 0,
    "Entropy": 0,
    "Neighbours": 0,
    "Membrane Thickness": 0,
    "Membrane Patch Size": 19,
    "Minimum Sigma": 0.5,
    "Maximum Sigma": 16,
}


# %% FUNCTIONS
def create_label_mask(
    img_h: int,
    img_w: int,
    arr_label_region: Polygon | Rectangle,
    label_val: int,
    label_type: LabelType,
    label_width: float = 0,
) -> np.ndarray:
    """Given a polygonal region, create a masked array filled with the class value by drawing polygon onto ImageDraw and converting to numpy."""
    temp_img = Image.new("L", (img_w, img_h), 0)
    if label_type in ["Polygon", "SAM"]:
        ImageDraw.Draw(temp_img).polygon(
            arr_label_region, outline=label_val, fill=label_val
        )
    elif label_type == "Rectangle":
        arr_label_region = cast(Rectangle, arr_label_region)
        ImageDraw.Draw(temp_img).rectangle(
            arr_label_region, outline=label_val, fill=label_val
        )
    elif label_type == "Circle":
        ImageDraw.Draw(temp_img).ellipse(
            arr_label_region, outline=label_val, fill=label_val
        )
    elif label_type in ["Brush", "Eraser"]:
        # if brush thin then draw width 1 lines connecting points
        if int(label_width) == 1:
            ImageDraw.Draw(temp_img).line(
                arr_label_region, fill=label_val, width=int(label_width)
            )
        else:  # if thick, draw circle @ every point (slower)
            arr_label_region = cast(Polygon, arr_label_region)
            for x, y in arr_label_region:
                top_left = (x - label_width, y - label_width)
                bottom_right = (x + label_width, y + label_width)
                ImageDraw.Draw(temp_img).ellipse(
                    [top_left, bottom_right], fill=label_val
                )
    else:
        raise Exception(f"Invalid labeling type - {label_type}")
    mask = np.array(temp_img)
    return mask


def resize_longest_side(img: Image.Image, l: int, patch_size: int = 14) -> Image.Image:
    oldh, oldw = img.height, img.width
    scale = l * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    neww = int(neww + 0.5)
    newh = int(newh + 0.5)
    neww = neww - (neww % patch_size)
    newh = newh - (newh % patch_size)

    return img.resize((neww, newh))


def get_training_data(
    feature_stack: np.ndarray, labels: np.ndarray, method="cpu"
) -> Tuple[np.ndarray, np.ndarray]:
    """Given $feature_stack and $labels, flatten both and reshape accordingly. Add a class offset if using XGB gpu."""
    h, w, feat = feature_stack.shape
    flat_labels = labels.reshape((h * w))
    flat_features = feature_stack.reshape((h * w, feat))
    labelled_mask = np.nonzero(flat_labels)

    fit_data = flat_features[labelled_mask[0], :]
    target_data = flat_labels[labelled_mask[0]]
    if method == "gpu":
        target_data -= 1
    return fit_data, target_data


# %% CLASSES
@dataclass
class Label:
    """A label with integer class value, a list of points forming a polygon (or rectangle/circle/brush strokes) and a label type."""

    class_value: int
    points: Polygon
    label_type: LabelType
    width: float = 0


@dataclass
class Piece:
    """
    Piece.

    Fundamental unit of program. Holds the data associated with the image in $arr, a PIL image of the arr in $img.
    Labels is a list of label objects that belong to this piece. __post_init__ adds some mutable objects like
    segmentations and grid_points that are useful later.
    """

    arr: np.ndarray
    img: Image.Image
    labels: List[Label]
    labelled: bool = False
    segmented: bool = False

    def __post_init__(self) -> None:
        """Set these here because dataclasses don't like mutable objects being assigned in __init__."""
        shape: Tuple[int, ...] = self.arr.shape[:-1]
        self.h: int = shape[0]
        self.w: int = shape[1]

        # integer arr where 0 = not labelled and N > 0 indicates a label for class N at that pixel
        self.labels_arr: np.ndarray = np.zeros(shape, dtype=np.uint8)
        # integer arr where value N at pixel P indicates the classifier thinks P is class N
        self.seg_arr: np.ndarray = np.zeros(shape, dtype=np.uint8)

        # boolean arr where 1 = show this pixel in the overlay and 0 means hide. Used for hiding/showing labels later.
        self.label_alpha_mask = np.ones_like(self.seg_arr, dtype=bool)

        self.features = np.zeros((self.h, self.w, 384))

    def _label_to_mask_arr(
        self,
        fractional_points: Polygon,
        class_value: int,
        labelling_type: LabelType,
        label_width: float = 0,
    ) -> np.ndarray:
        """Given list of fractional points, scale with img's w and h then create label mask."""
        scaled_points: Polygon = []
        for p in fractional_points:
            scaled_points.append(((p[0] * self.w), (p[1] * self.h)))

        added_mask: np.ndarray = create_label_mask(
            self.h, self.w, scaled_points, class_value, labelling_type, label_width
        )
        return added_mask

    def add_label_to_mask(self, label: Label) -> None:
        """Given a label object, add it to $mask which is an arr the same size as $arr but 0 for unlabelled regions and $classvalue for labelled regions."""
        added_label_arr = self._label_to_mask_arr(
            label.points, label.class_value, label.label_type, label.width
        )
        # add check here if erasing - maybe abstract this update into a function
        # everywhere that added_label_arr is set and isn't already labelled, set labels_arr to that value
        if label.class_value == 255:  # erasing
            self.labels_arr = np.where(
                added_label_arr == 255, 0, self.labels_arr
            ).astype(np.uint8)
        else:
            self.labels_arr = np.where(
                added_label_arr > 0,
                added_label_arr,
                self.labels_arr,
            ).astype(np.uint8)
        self.label_alpha_mask = np.where(self.labels_arr > 0, True, False).astype(bool)

        if label.class_value != 255:
            self.labels.append(label)
        self.labelled = True

    def remove_label_by_index(self, index: int) -> None:
        """Remove label at $index then redraw the piece's labels arr from scratch."""
        self.labels.pop(index)
        self.labels_arr = 0 * self.labels_arr
        for label in self.labels:
            self.add_label_to_mask(label)


class DataModel:
    """
    DataModel.

    Holds every Piece during runtime, manages communication to threaded classifiers via queues. Interfaces with
    GUI and has some post processing functions.
    """

    def __init__(self) -> None:
        """Initialise all values."""
        self.gallery: List[Piece] = []

        self.current_piece: Piece
        self.current_piece_idx: int = 0

        self.labelling_type: LabelType = "Polygon"

        self.current_class: int = 1

        self.send_queue: Queue = Queue(maxsize=40)
        self.recv_queue: Queue = Queue(maxsize=40)

        self.worker: Process

        self.get_net("dinov2_vits14_reg")
        # our classifier's queues are swapped relative to data model

    def get_net(self, model_name: str = "dinov2_vits14_reg", stride: int = 4):
        self.net = HighResDV2(model_name, stride, pca_dim=-1, dtype=16)
        self.net.cuda()
        self.net.eval()

        shift_dists = [i for i in range(1, 2)]
        fwd_shift, inv_shift = tr.get_shift_transforms(shift_dists, "Moore")
        self.net.set_transforms(fwd_shift, inv_shift)

        # self.net = torch.hub.load("mhamilton723/FeatUp", "dinov2", use_norm=False)
        # fwd_flip, inv_flip = tr.get_flip_transforms()
        # fwd, inv = tr.combine_transforms(fwd_shift, fwd_flip, inv_shift, inv_flip)

    def load_image_from_filepath(self, filepath: str) -> Image.Image:
        """Given a filepath, either: load array then create image or load image then create array (depending on extension)."""
        extension: str = filepath.split(".")[-1]
        if extension.lower() not in ["png", "jpg", "jpeg", "tif", "bmp", "tiff"]:
            raise Exception(f".{extension} is not a valid image file format")

        pil_image: Image.Image
        np_array: np.ndarray  # between 0 and 255
        if extension.lower() in ["tiff", "tif"]:
            np_array: np.ndarray = imread(filepath)  # type: ignore
            np_array = (np_array / np.amax(np_array)) * 255
            np_array = np_array.astype(np.uint8)
            new_shape = [1 for i in np_array.shape] + [3]
            np_array = np.expand_dims(np_array, -1)
            np_array = np.tile(np_array, new_shape)
            pil_image = Image.fromarray(np_array)
            pil_image = resize_longest_side(pil_image, 322)
            np_array = np.array(pil_image)
            pil_image = pil_image.convert("RGBA")
        else:  # done s.t data channel is 1-d. fix later
            pil_image = Image.open(filepath).convert("RGB")
            np_array = np.asarray(pil_image)
            np_array = (np_array / np.amax(np_array)) * 255
            pil_image = resize_longest_side(pil_image, 322)
            np_array = np.array(pil_image)
            pil_image = pil_image.convert("RGBA")

        new_piece: Piece = Piece(np_array, pil_image, [])
        self.gallery.append(new_piece)
        self.current_piece = new_piece

        # for new image, get sam encoding and store
        self._get_features(new_piece)

        return pil_image

    def _get_features(self, piece: Piece) -> None:
        np_array = piece.arr
        rgb_pil_img = Image.fromarray(np_array)
        tensor: torch.Tensor = tr.to_norm_tensor(rgb_pil_img)
        tensor = tensor.cuda()
        # tensor = tensor.unsqueeze(0)
        feats = self.net.forward_sequential(tensor)
        feats = interpolate(feats, (rgb_pil_img.height, rgb_pil_img.width))
        feats_np = tr.to_numpy(feats)
        piece.features = feats_np.transpose((1, 2, 0))

        # self.sam_predictor.set_image(rgb_arr)
        # if self.current_piece is not None:
        #    self.current_piece.sam_encoding = self.sam_predictor.features

    # def _get_features(self, img: )

    def set_current_img(self, x: int) -> Image.Image:
        """Given index x (i.e from slider), update attrs and return corresponding PIL image."""
        self.current_piece_idx = x
        self.current_piece = self.gallery[x]
        # update sam encoding
        # self.sam_predictor.features = self.current_piece.sam_encoding
        return self.current_piece.img

    def _set_current_class(self, class_val: int) -> None:
        self.current_class = class_val

    def add_label(
        self,
        label_class: int,
        label_region: Polygon,
        label_type: LabelType,
        label_width: float = 0,
    ) -> None:
        """Add label to the datamodel - usually called from draw_polygon_canvas on_click event."""
        current_class = label_class if label_type != "Eraser" else 255
        label: Label = Label(current_class, label_region, label_type, label_width)
        if self.current_piece is not None:
            self.current_piece.add_label_to_mask(label)

    def train(self):
        """Start a thread training & applying the Random Forest classifier."""
        imgs = [piece.arr for piece in self.gallery]
        labels = [piece.labels_arr for piece in self.gallery]
        feats = [piece.features for piece in self.gallery]

        init = False
        for i, (label, feat) in enumerate(zip(labels, feats)):
            if self.gallery[i].labelled == False:
                continue

            if init == False:
                fit_data, target_data = get_training_data(feat, label)
                all_fit_data = fit_data  # normalise_pca(fit_data)
                all_target_data = target_data
                init = True
            else:
                fit_data, target_data = get_training_data(feat, label)
                # fit_data = normalise_pca(fit_data)

                all_fit_data = np.concatenate((all_fit_data, fit_data), axis=0)
                all_target_data = np.concatenate((all_target_data, target_data), axis=0)

        model = LogisticRegression("l2", n_jobs=12, max_iter=1000, warm_start=True)
        model.fit(all_fit_data, all_target_data)
        print("done")

        for piece in self.gallery:
            feats = piece.features
            h, w, c = feats.shape
            flat_features = feats.reshape((h * w, c))
            flat_feats_norm = normalise_pca(flat_features)

            flat_classes = model.predict(flat_features)
            piece.seg_arr = flat_classes.reshape((h, w))
            piece.segmented = True
        self.send_queue.put("test")

    def _finish_worker_thread(self):
        self.worker.terminate()
