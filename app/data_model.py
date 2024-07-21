"""Bread and butter of the app. Holds all the images and numpy array data for each image we want to segment."""

# %% IMPORTS
import numpy as np
from PIL import Image, ImageDraw
from multiprocessing import Process, Queue, set_start_method
from math import floor

from tifffile import imread, imwrite

from dataclasses import dataclass
from typing import TypeAlias, List, Tuple, Literal, cast

from classifiers import Model, get_featuriser_classifier

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

        self.threaded = False

        self.send_queue: Queue = Queue(maxsize=40)
        self.recv_queue: Queue = Queue(maxsize=40)

        self.worker: Process

        self.selected_model: Literal["DINOv2-S-14", "DINO-S-8", "RandomForest"] = (
            "DINOv2-S-14"
        )

        self.get_model(self.selected_model)

    def get_model(self, model_name: str) -> None:
        # our classifier's queues are swapped relative to data model
        self.model = get_featuriser_classifier(
            model_name, self.recv_queue, self.send_queue
        )
        if model_name != self.selected_model:
            self.selected_model = model_name  # type: ignore
            self.get_features()

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
            pil_image = resize_longest_side(pil_image, 518)  # 322 \518
            np_array = np.array(pil_image)
            pil_image = pil_image.convert("RGBA")
        else:  # done s.t data channel is 1-d. fix later
            pil_image = Image.open(filepath).convert("RGB")
            np_array = np.asarray(pil_image)
            np_array = (np_array / np.amax(np_array)) * 255
            pil_image = resize_longest_side(pil_image, 518)
            np_array = np.array(pil_image)
            pil_image = pil_image.convert("RGBA")

        new_piece: Piece = Piece(np_array, pil_image, [])
        self.gallery.append(new_piece)
        self.current_piece = new_piece
        return pil_image

    def save_seg(self, file_obj) -> None:
        seg_arr = self.current_piece.seg_arr.astype(np.uint8)
        max_class = np.amax(seg_arr)
        delta = floor(255 / (max_class))
        rescaled = ((seg_arr * delta)).astype(np.uint8)

        imwrite(file_obj.name, rescaled, photometric="minisblack")

    def set_current_img(self, x: int) -> Image.Image:
        """Given index x (i.e from slider), update attrs and return corresponding PIL image."""
        self.current_piece_idx = x
        self.current_piece = self.gallery[x]
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

    def get_features(self) -> None:
        imgs = [piece.img for piece in self.gallery]
        inds = [i for i in range(len(self.gallery))]
        if self.threaded:
            self.worker = Process(target=self.model.get_features, args=(imgs, True))
            self.worker.start()
        else:
            features = self.model.get_features(imgs, inds, False)
            for i, f in enumerate(features):
                self.gallery[i].features = f

    def train(self) -> None:
        """Start a thread training & applying classifier."""
        labels = [piece.labels_arr for piece in self.gallery if piece.labelled]
        feats = [piece.features for piece in self.gallery if piece.labelled]

        if self.threaded:
            self.worker = Process(target=self.model.train, args=(feats, labels, True))
            self.worker.start()
        else:
            self.model.train(feats, labels, False)
            self.segment()
        print("done")

    def segment(self) -> None:
        feats = [piece.features for piece in self.gallery]
        imgs = [piece.img.convert("RGB") for piece in self.gallery]
        inds = [i for i in range(len(self.gallery))]
        if self.threaded:
            self.worker = Process(target=self.model.segment, args=(feats, inds, True))
            self.worker.start()
        else:
            segmentations = self.model.segment(feats, imgs, inds, False)
            for i, s in enumerate(segmentations):
                self.gallery[i].seg_arr = s
                self.gallery[i].segmented = True
            self.recv_queue.put({"test": "_"})

    def _finish_worker_thread(self):
        self.worker.terminate()
