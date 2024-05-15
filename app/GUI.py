"""Displays Tkinter app with nice ttk styling. Also contains scheduler/listener for classifier thread."""

# %% IMPORTS
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd
from gui_elements._small_widgets import HighlightButton, TextMenuFrame
from gui_elements.zoom_scroll_canvas import CanvasImage
from matplotlib.colors import ListedColormap
from time import time
import pickle
from os import getcwd

import numpy as np
from data_model import DataModel, LabelType, Label


# this is a silly/funny/genius hack from stackoverflow to use the tooltips from IDLE (as tkinter doesn't have its own)
from idlelib.tooltip import Hovertip
from PIL import Image, ImageTk
from gui_elements.zoom_scroll_canvas import CanvasImage

from typing import Literal, List, Callable, Tuple, Any, Union, cast
from functools import wraps

# %% CONSTANTS
CANVAS_W: int = 850
CANVAS_H: int = 1200
CANVAS_H_GRID: int = 5
CANVAS_W_GRID: int = 1

PAD: int = 10
MENU_BAR_ROW: int = 0
SIDE_COL: int = 2

MIN_TIME: float = 0.07


# %% FUNCTIONS
def _foo(x):  # placeholder fn to be deleted later
    print("Not implemented")


def _foo_n(x):
    pass


def _no_arg_foo():
    print("Not implemented")


def _make_frame_contents_expand(frame: tk.Tk | tk.Frame | ttk.LabelFrame, i=5):
    for idx in [0, 1, 2, 3, 4]:
        frame.columnconfigure(index=idx, weight=1)
        frame.rowconfigure(index=idx, weight=1)


def open_file_dialog_return_fps(
    title: str = "Open",
    file_type_name: str = "Image",
    file_types_string: str = ".tif .tiff .png .jpg",
) -> Union[Literal[""], Tuple[str, ...]]:
    """Open file dialog and select n files, returning their file paths then loading them."""
    filepaths: Union[Literal[""], Tuple[str, ...]] = fd.askopenfilenames(
        filetypes=[(f"{file_type_name} files:", file_types_string)], title=title
    )

    if filepaths == ():  # if user closed file manager w/out selecting
        return ""
    return filepaths


# %% =================================== MAIN APP ===================================
class App(ttk.Frame):
    """Parent widget for GUI. Contains event scheduler in listen() method."""

    def __init__(self, root: tk.Tk, data_model: DataModel) -> None:
        """Take $root and assign it to attr .root. Inits other widgets and starts scheduler."""
        ttk.Frame.__init__(self)
        self.root = root
        self.data_model = data_model

        # boolean flag if overlays need to be updated
        self.needs_updating: bool = False
        self.selected_overlay: Literal["Segmentation", "Label"] = "Segmentation"
        self.seg_alpha: float = 1.0
        self.label_alpha: float = 0.7
        self.brush_width: float = 1.0

        self.class_colours = [
            "#fafafa",
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]
        self.cmap = ListedColormap(self.class_colours)

        self.root.option_add("*tearOff", False)
        _make_frame_contents_expand(self.root)

        self._init_ttk_theme()
        self._init_pack_widgets()

        self.event_loop()

    # ADD WIDGETS
    def _init_ttk_theme(self, theme: Literal["light", "dark"] = "light") -> None:
        self.root.tk.call(
            "source", f"{getcwd()}/samba/gui_elements/tk_themes/azure.tcl"
        )
        self.root.tk.call("set_theme", theme)
        # needed to stop file dialog font being white (and therefore unreadable)
        self.root.option_add("*TkFDialog*foreground", "black")

    def _init_pack_widgets(self) -> None:
        self.menu_bar = MenuBar(self.root, self)
        self.root.config(menu=self.menu_bar)

        img_frame = ttk.LabelFrame(self, text="Image", padding=(3.5 * PAD, 3.5 * PAD))
        img_frame.grid(
            row=MENU_BAR_ROW + 1,
            column=0,
            padx=(2 * PAD, PAD),
            pady=(2 * PAD, PAD),
            rowspan=CANVAS_H_GRID,
            columnspan=CANVAS_W_GRID,
            sticky="nsew",
        )
        img_frame.rowconfigure(0, weight=1, minsize=CANVAS_W)
        img_frame.columnconfigure(0, weight=1, minsize=CANVAS_H)
        self.canvas = PolygonCanvas(img_frame, self)
        self.canvas.grid(row=0, column=0)

        self.train_button = ttk.Button(
            self,
            text="Train Classifier!",
            command=self.data_model.train,
        )
        self.train_button.grid(row=MENU_BAR_ROW + 1, column=SIDE_COL, pady=(3 * PAD, 0))

        self._init_labels_frame()
        self._init_overlays_frame()
        self._init_treeview()
        self._init_slider()

    def _init_labels_frame(self) -> None:
        labels_frame = ttk.LabelFrame(self, text="Labels")
        labels_frame.grid(
            row=MENU_BAR_ROW + 2,
            column=SIDE_COL,
            padx=(PAD, PAD),
            pady=(PAD, PAD),
            sticky="ew",
        )

        def change_type(label_type: LabelType) -> None:
            print(label_type)
            self.data_model.labelling_type = label_type
            self.canvas.cancel(None)

        def toggle_eraser() -> None:
            pass

        # TODO: create eraser by setting the class val to 0.1 (this will)
        # First row: different label/brush types
        label_type_frame = tk.Frame(labels_frame)
        # in format icon path, fn, tooltip text
        dir_prefix = f"{getcwd()}/samba/gui_elements/icons/"
        label_type_list: List[Tuple[str, Callable, str]] = [
            (
                f"{dir_prefix}smart.png",
                lambda: change_type("SAM"),
                "Smart Labelling",
            ),
            (
                f"{dir_prefix}polygon.png",
                lambda: change_type("Polygon"),
                "Polygon",
            ),
            (f"{dir_prefix}brush.png", lambda: change_type("Brush"), "Brush"),
            (
                f"{dir_prefix}rectangle.png",
                lambda: change_type("Rectangle"),
                "Rectangle",
            ),
            (f"{dir_prefix}circle.png", lambda: change_type("Circle"), "Circle"),
            (
                f"{dir_prefix}erase.png",
                lambda: change_type("Polygon"),
                "Toggle eraser",
            ),
        ]
        self.btn_imgs = []
        for i, label_type_tuple in enumerate(label_type_list):
            pad: int = 8 if i == len(label_type_list) - 1 else 2
            photo_dir, command, tooltip_txt = label_type_tuple
            # Fancy toggleable button element (frame holding a button with an icon)
            highlight_btn = HighlightButton(label_type_frame)
            photoimage = highlight_btn._get_photoimage(photo_dir)
            # need to save (and refer to) permanent reference to image to stop it being garbage collected (hence the attr)
            self.btn_imgs.append(photoimage)
            highlight_btn._init_button(self.btn_imgs[i], command, tooltip_txt)
            highlight_btn.grid(column=i, row=0, padx=(pad, pad))
            if i == 0:
                self._temp_highlight_btn = highlight_btn
                highlight_btn._toggle_btn_and_call_fn(highlight_btn, command)

        label_type_frame.grid(row=0, column=0, padx=(PAD, PAD), pady=(PAD, PAD))

        # Second row: brush width
        def set_brush_width(x):
            self.brush_width = float(x)

        brush_frame = TextMenuFrame(
            labels_frame,
            set_brush_width,
            "Brush Width:",
            "slider",
            [1.0, 64.0],
        )
        brush_frame.grid(row=1, column=0, padx=(PAD, PAD), pady=(PAD, PAD))
        # Third row: save/load labels
        label_io_frame = tk.Frame(labels_frame)
        save_btn = ttk.Button(
            label_io_frame, text="Save", command=self.save_labels, width=4
        )
        load_btn = ttk.Button(
            label_io_frame, text="Load", command=self._load_labels_from_fd, width=4
        )
        for i, btn in enumerate([save_btn, load_btn]):
            btn.grid(row=0, column=i, padx=(5, 5))
        label_io_frame.grid(row=2, column=0, padx=(PAD, PAD), pady=(PAD, PAD))

    def _init_overlays_frame(self) -> None:
        overlays_frame = ttk.LabelFrame(self, text="Overlays")
        overlays_frame.grid(
            row=MENU_BAR_ROW + 3,
            column=SIDE_COL,
            padx=(PAD, PAD),
            pady=(PAD, PAD),
            sticky="ew",
        )

        # First row: opacity slider for selected frame
        def change_opacity(x):
            val = float(x)
            if self.selected_overlay == "Segmentation":
                self.seg_alpha = val
            else:
                self.label_alpha = val
            self.needs_updating = True

        opacity_frame = TextMenuFrame(
            overlays_frame, lambda x: change_opacity(x), "Opacity:", "slider"
        )
        # init slider with right value (need to cast bc textmenuframe has 2 possible widgets)
        slider = cast(ttk.Scale, opacity_frame.menu)
        slider.set(self.seg_alpha)
        opacity_frame.grid(row=0, column=0, padx=(PAD, PAD), pady=(PAD, PAD))

        # Second row: overlay type selection
        def change_overlay_type(x):
            if self.selected_overlay == "Segmentation":
                val = self.seg_alpha
            else:
                val = self.label_alpha
            slider.set(val)
            self.selected_overlay = x

        overlay_type = TextMenuFrame(
            overlays_frame,
            change_overlay_type,
            "Type:",
            "dropdown",
            ["Segmentation", "Label"],
        )
        overlay_type.grid(row=1, column=0, padx=(PAD, PAD), pady=(PAD, PAD))

    def _init_treeview(self) -> None:
        self.treeview = ScrollableTreeview(self)
        self.treeview.grid(
            row=MENU_BAR_ROW + 4,
            column=SIDE_COL,
            padx=(PAD, PAD),
            pady=(0, 0),
            sticky="nsew",
        )
        treeview_buttons_frame = tk.Frame(self)
        add_btn = ttk.Button(
            treeview_buttons_frame, text="Add Class", command=_no_arg_foo, width=12
        )
        remove_btn = ttk.Button(
            treeview_buttons_frame, text="Remove Class", command=_no_arg_foo, width=12
        )
        for i, btn in enumerate([add_btn, remove_btn]):
            btn.grid(row=0, column=i, padx=(PAD, PAD))
        treeview_buttons_frame.grid(row=5, column=2, sticky="ew", padx=(PAD, PAD))
        self.treeview._change_colour(self.class_colours[1])

    def _init_slider(self, to: int = 0, set_to: int = 0) -> None:
        def slider_fn(x):
            x = int(float(x))
            img = self.data_model.set_current_img(x)
            self.canvas.set_current_image(img)

        self.slider = ttk.Scale(self, from_=0, to=to, command=slider_fn)
        if to == 0:  # early return
            return None
        self.slider.grid(
            row=MENU_BAR_ROW + CANVAS_H_GRID + 1,
            column=0,
            columnspan=1,
            sticky="ew",
            padx=(3 * PAD, 3 * PAD),
            pady=(0, PAD),
        )
        self.slider.set(set_to)

    # LOGIC

    def load_image_from_filepaths(self, image_paths: Tuple[str, ...]) -> None:
        """Given a list of file pahs (i.e from file dialog), ask data model to load them then display last one."""
        for img_path in image_paths:
            temp_img = self.data_model.load_image_from_filepath(img_path)
        self.canvas.image_available = True
        self.canvas.set_current_image(temp_img, new=True)

        n_imgs_disp: int = len(self.data_model.gallery) - 1
        self._init_slider(to=n_imgs_disp, set_to=n_imgs_disp)  # update slider
        # self.needs_updating = True
        self.canvas.canvas.focus_set()

    def set_label_class(self, class_val: int) -> None:
        """Set current class on data model, update highlight colours and chnage selected class in treeview."""
        self.data_model._set_current_class(class_val)
        selected_class_colour = self.class_colours[class_val]
        self.canvas.fill_colour = selected_class_colour
        self._temp_highlight_btn._change_prev_btn_colour(selected_class_colour)
        self.treeview._change_colour(selected_class_colour)
        self.treeview._add_new_class(class_val)

    def save_labels(self) -> None:
        """Save labels for current piece to pickle file."""
        model = self.data_model
        with open(f"img_{model.current_piece_idx}_labels.pkl", "wb") as f:
            pickle.dump(
                list(model.current_piece.labels), f, protocol=pickle.HIGHEST_PROTOCOL
            )

    def _load_labels_from_fd(self) -> None:
        labels_files = open_file_dialog_return_fps("Load labels", "pickle", ".pkl")
        self.load_labels_from_fp(labels_files[0])

    def load_labels_from_fp(self, fp) -> None:
        """Load list of labels and add them to $data_model.current_piece."""
        model = self.data_model
        with open(f"{fp}", "rb") as f:
            labels = pickle.load(f)
        for label in labels:
            model.current_piece.add_label_to_mask(label)
        self.treeview._update_treeview_from_labels(self.data_model.current_piece.labels)
        # self.needs_updating = True

    def _load_segmentations(self, data: List[np.ndarray]) -> None:
        class_offset = 1
        for i in range(len(data)):
            piece = self.data_model.gallery[i]
            probs = data[i]
            classes = np.argmax(probs, axis=0).astype(np.uint8)
            piece.seg_arr = classes + class_offset
            piece.segmented = True

    # DISPLAY
    def get_img_from_seg(
        self, train_result: np.ndarray, cmap: ListedColormap, alpha_mask: np.ndarray
    ) -> Image.Image:
        """Given a segmentation (i.e H,W arr where entries are ints), map this using the colourmaps to an image (with set opacity)."""
        cmapped = cmap(train_result)
        cmapped[:, :, 3] = alpha_mask
        cmapped = (cmapped * 255).astype(np.uint8)
        pil_image = Image.fromarray(cmapped, mode="RGBA")
        return pil_image

    def show_overlay(self) -> None:
        """
        show_overlay.

        Given a segmentation and/or labels, create image with opacity given by the respective seg/label opacity
        float, then paste it on top of the data image.
        """
        self.needs_updating = False
        if len(self.data_model.gallery) == 0:  # early return if no data
            return None
        current_piece = self.data_model.current_piece
        arr_shape: Tuple[int, int] = (current_piece.h, current_piece.w)

        new_img = Image.new(size=arr_shape[::-1], mode="RGBA")
        new_img.paste(current_piece.img, (0, 0), current_piece.img)
        if current_piece.segmented is True:
            seg_data = current_piece.seg_arr
            alpha_mask = np.ones_like(seg_data, dtype=np.float16) * self.seg_alpha
            overlay_seg_img = self.get_img_from_seg(
                seg_data, cmap=self.cmap, alpha_mask=alpha_mask
            )
            new_img.paste(overlay_seg_img, (0, 0), overlay_seg_img)

        if current_piece.labelled is True:
            label_data = current_piece.labels_arr
            alpha_mask = (
                np.where(label_data > 0, 1, 0).astype(np.float16) * self.label_alpha
            )
            overlay_label_img = self.get_img_from_seg(
                label_data, cmap=self.cmap, alpha_mask=alpha_mask
            )
            new_img.paste(overlay_label_img, (0, 0), overlay_label_img)

        self.canvas.set_current_image(new_img)

    def _map_data_header_to_action(
        self, data_header: str, data: List[np.ndarray]
    ) -> None:
        if data_header == "segmentation":
            self._load_segmentations(data)
            self.data_model._finish_worker_thread()
        else:
            raise Exception("Queue return type not implemented.")

    def event_loop(self) -> None:
        """
        event_loop.

        Start a scheduler to check the data queue every 100ms. If it gets some data, for each piece that has data sent back,
        read the segmentation and assign it to the piece in the data model (including probabilities, uncertainties etc.).
        """
        # self.canvas.canvas.focus_set()
        queue = self.data_model.recv_queue
        while queue.empty() is False:
            data_in = queue.get_nowait()

            for data_header, data in data_in.items():
                self._map_data_header_to_action(data_header, data)
            self.needs_updating = True

        if self.needs_updating:
            self.show_overlay()
        self.loop = self.root.after(100, self.event_loop)


# %% =================================== MENUBAR/TREEVIEW ===================================


class MenuBar(tk.Menu):
    """Menu bar across top of GUI with dropdown commands: load data, classifiers, save segs etc."""

    def __init__(self, root: tk.Tk, app: App) -> None:
        """Attach to root then initialise all the sub menus: data, classifiers, post process & save."""
        super(MenuBar, self).__init__(
            root
        )  # done s.t the menu bar is attached to the root (tk window) rather than the frame
        self.app = app

        data_name_fn_pairs: List[Tuple[str, Callable]] = [
            ("Add Image", self._load_images),
            ("Remove Image", _foo),
        ]
        data_menu = self._make_dropdown(data_name_fn_pairs)
        self.add_cascade(label="Data", menu=data_menu)

        classifier_name_fn_pairs: List[Tuple[str, Callable]] = [
            ("New Classifier", _foo),
            ("Train Classifier", _foo),
            ("Apply Classifier", _foo),
            ("Load Classifier", _foo),
            ("Save Classifier", _foo),
            ("sep", _foo),
            ("Features", _foo),
        ]
        classifier_menu = self._make_dropdown(classifier_name_fn_pairs)
        self.add_cascade(label="Classifier", menu=classifier_menu)

        self.add_command(label="Post-Process", command=_foo)  # type: ignore
        self.add_command(label="Save Segmentation", command=_foo)  # type: ignore

    def _make_dropdown(self, name_fn_pair_list: List[Tuple[str, Callable]]) -> tk.Menu:
        menu = tk.Menu()
        n_commands: int = len(name_fn_pair_list)
        for i in range(n_commands):
            name, function = name_fn_pair_list[i]
            if name == "sep":
                menu.add_separator()
            else:
                menu.add_command(label=name, command=function)
        return menu

    def _load_images(self) -> None:
        file_paths = open_file_dialog_return_fps(title="Load Images")
        if file_paths == "":  # user closed fd or selected no files
            pass
        else:
            self.app.load_image_from_filepaths(file_paths)


class ScrollableTreeview(tk.Frame):
    """ScrollableTreeview on left hand side that shows classes and their associated labels."""

    # TODO: make this frame background change colour when different class selected
    def __init__(self, parent: App) -> None:
        """Init scrollbar, treeview and bind mouse events."""
        super().__init__(parent)
        scrollbar = ttk.Scrollbar(self)
        scrollbar.pack(side="right", fill="both")
        self.app = parent

        self.treeview = ttk.Treeview(
            self,
            selectmode="browse",
            yscrollcommand=scrollbar.set,
            columns=("1", "2"),
        )
        self.treeview.pack(expand=True, fill="both", padx=(3, 3), pady=(3, 3))
        scrollbar.config(command=self.treeview.yview)

        self.treeview.column("#0", anchor="w", width=75)
        self.treeview.heading("#0", text="Labels", anchor="center")
        self.treeview.column(1, anchor="w", width=80)
        self.treeview.heading(1, text="Vol. Fracs", anchor="center")
        # Ugly hack - this thing has 3 columns even though I've only defined 2 so make it's width smaller
        self.treeview.column(2, anchor="w", width=1)

        self.data: List[List[Any]] = [["", i, f"Class {i}"] for i in range(1, 3)]
        self.n_classes = 2
        self.reset_treeview_add_new_data(self.data)

        self.treeview.bind("<ButtonRelease-1>", self._treeview_after_click)

    def _change_colour(self, colour: str) -> None:
        self["bg"] = colour
        self["relief"] = "sunken"

    def reset_treeview_add_new_data(self, data: List[List[Any]]) -> None:
        """Reset a treeview and replace old data with $data."""
        tree = self.treeview
        tree.delete(*tree.get_children())
        self.n_classes = 0
        for item in data:
            # adding new class
            if item[0] == "":
                value = ""
                self.n_classes += 1
            else:
                value = ""  # item[3]
            tree.insert(
                parent=item[0],
                index="end",
                iid=item[1],
                text=item[2],
                values=value,  # type: ignore
                tags=str(item[1]),
            )
            if item[0] == "" or item[1] in {8, 21}:
                tree.item(item[1], open=True)
        self.data = data

    def _add_new_class(self, class_val: int) -> None:
        if self.n_classes < class_val:
            for i in range((self.n_classes + 1), (class_val + 1)):
                self.data.append(["", len(self.data) + 1, f"Class {i}"])
                self.reset_treeview_add_new_data(self.data)
            self.n_classes = class_val

    def _treeview_after_click(self, event) -> None:
        """Once treeview is clicked set that class as active. TODO: set label as active as well so can delete."""
        item = self.treeview.selection()
        data = self.data[int(item[0]) - 1]
        if data[0] == "":
            class_n = int(data[2][-1])
            self.app.set_label_class(class_n)

    def _labels_to_treeview_data(self, labels: List[Label]) -> List[List[Any]]:
        tree_data: List[List[Any]] = []
        label_data: List[int] = [0 for i in range(self.n_classes)]
        for label in labels:
            label_data[label.class_value - 1] += 1

        global_count = 0
        parent_idx = 0
        for i, label_count in enumerate(label_data):
            class_entry = [
                "",
                global_count,
                f"Class {i + 1}",
            ]
            tree_data.append(class_entry)
            parent_idx = global_count
            global_count += 1
            for j in range(label_count):
                tree_entry = [parent_idx, global_count, f"Label {j}"]
                tree_data.append(tree_entry)
                global_count += 1
        return tree_data

    def _update_treeview_from_labels(self, labels: List[Label]) -> None:
        tree_data = self._labels_to_treeview_data(labels)
        self.reset_treeview_add_new_data(tree_data)


# %% =================================== CANVAS ===================================


class PolygonCanvas(CanvasImage):
    """
    PolygonCanvas.

    Inherits from gui_elements/zoom_scroll_canvas.py. Contains all the methods for drawing onto the
    zooming/scrolling canvas and passing that data to the GUI. Generally the pattern is: draw on
    canvas object then once labe lis confirmed, send to data model, delete drawing on canvas,
    update label overlay with the confirmed data.
    """

    def __init__(self, parent: tk.Widget, app: App):
        """Init the canvas and bind all the keypresses."""
        super(PolygonCanvas, self).__init__(parent)
        self.app = app

        self.image_available = False

        self.prev_t = time()
        self.fill_colour: str = self.app.class_colours[1]
        self.current_label_frac_points: List[Tuple[float, float]] = []

        self.canvas.bind("<Button-1>", self.left_click)
        self.canvas.bind("<Button-3>", self.right_click)

        self.canvas.bind("<Motion>", self.mouse_motion)
        self.canvas.bind("<B1-Motion>", self.mouse_motion_while_click)
        self.canvas.bind("<ButtonRelease-1>", self.mouse_release)

        self.canvas.bind("<Escape>", self.cancel)
        self.canvas.bind("<Delete>", self.delete)

        for i in range(10):
            self.canvas.bind(f"{i}", self._num_key_press)

    def left_click(self, event: tk.Event) -> None:
        """On left click, find out label type then pass event x, y into corresponding method."""
        # TODO: figure out how to make this result -> bounds check -> fn pattern a decorator
        result = self._bounds_check_return_coords(event)
        if result is None:
            return None
        else:
            model = self.app.data_model
            if model.labelling_type == "Polygon":
                self.place_poly_point(*result)
            # elif model.labelling_type == "SAM":
            # self._place_sam_label(event)
            else:
                return None

    def right_click(self, event) -> None:
        """On right click, finish polygon label OR change SAM hypothesis index."""
        result = self._bounds_check_return_coords(event)
        if result is None:
            return None
        else:
            model = self.app.data_model
            if model.labelling_type == "Polygon":
                self.finish_poly(event)
            # elif model.labelling_type == "SAM":
            #    x, y, _, _ = result
            #    self._increment_sam_hypothesis_idx(x, y)
            else:
                return None

    def mouse_motion(self, event) -> None:
        """For live updating in progress polygon labels or SAM labelling."""

        if self.image_available is False:
            return

        result = self._bounds_check_return_coords(event)
        if result is None:
            return None
        else:
            is_polygon = self.app.data_model.labelling_type == "Polygon"
            is_sam = self.app.data_model.labelling_type == "SAM"
            current_t = time()
            enough_time_passed = current_t - self.prev_t > MIN_TIME
            points_placed = len(self.current_label_frac_points) > 0

            if is_polygon and points_placed:
                x, y, _, _ = result
                self._mouse_motion_poly(x, y)
            # elif is_sam and enough_time_passed:
            #    x, y, _, _ = result
            #    self._sam_suggest(x, y)

    def mouse_motion_while_click(self, event) -> None:
        """For brush type labelling."""
        result = self._bounds_check_return_coords(event)
        if result is None:
            return None
        else:
            if self.app.data_model.labelling_type == "Brush":
                self.place_poly_point(*result, int(self.app.brush_width))

    def mouse_release(self, event) -> None:
        """For brush/circle/rectangle label types (others covered by different click)."""
        result = self._bounds_check_return_coords(event)
        if result is None:
            return None
        else:
            if self.app.data_model.labelling_type == "Brush":
                self.finish_poly(event)

    def cancel(self, event) -> None:
        """On esc key, cancel current label and delete in progress drawings on canvas."""
        self.canvas.delete("in_progress")
        self.current_label_frac_points = []

    def delete(self, event) -> None:
        """Delete selected label (in treeview)."""
        _foo_n(event)

    def _num_key_press(self, event):
        number = int(event.char)
        print(number)
        self.app.set_label_class(number)

    # CONVERSION
    def _canvas_to_frac_coords(
        self, canvas_x: int, canvas_y: int
    ) -> Tuple[float, float]:
        bbox = self.canvas.coords(self.container)
        dx, dy = bbox[2] - bbox[0], bbox[3] - bbox[1]
        frac_x, frac_y = (canvas_x - bbox[0]) / dx, (canvas_y - bbox[1]) / dy
        return (frac_x, frac_y)

    def _canvas_to_arr_coords(self, canvas_x: int, canvas_y: int) -> Tuple[int, int]:
        h, w = self.app.data_model.current_piece.arr.shape
        frac_x, frac_y = self._canvas_to_frac_coords(canvas_x, canvas_y)
        return (int(frac_x * w), int(frac_y * h))

    def _frac_to_canvas_coords(
        self, frac_x: float, frac_y: float
    ) -> Tuple[float, float]:
        bbox = self.canvas.coords(self.container)
        dx, dy = bbox[2] - bbox[0], bbox[3] - bbox[1]
        canvas_x, canvas_y = (frac_x * dx) + bbox[0], (frac_y * dy) + bbox[1]
        return (canvas_x, canvas_y)

    def _arr_to_frac_coords(self, arr_x: int, arr_y: int) -> Tuple[float, float]:
        h, w = self.app.data_model.current_piece.arr.shape
        return arr_x / w, arr_y / h

    def _arr_to_canvas_coords(self, arr_x: int, arr_y: int) -> Tuple[float, float]:
        frac_x, frac_y = self._arr_to_frac_coords(arr_x, arr_y)
        canvas_x, canvas_y = self._frac_to_canvas_coords(frac_x, frac_y)
        return (canvas_x, canvas_y)

    # LOGIC
    def _bounds_check_return_coords(
        self, event: tk.Event
    ) -> Tuple[int, int, float, float] | None:
        bbox = self.canvas.coords(self.container)
        x, y = int(self.canvas.canvasx(event.x)), int(self.canvas.canvasy(event.y))
        if bbox[0] < x < bbox[2] and bbox[1] < y < bbox[3]:
            frac_x, frac_y = self._canvas_to_frac_coords(x, y)
            return x, y, frac_x, frac_y
        else:
            return None

    def place_poly_point(
        self, x: int, y: int, frac_x: float, frac_y: float, r: int = 5
    ) -> None:
        """Draw oval at click. Draw line from prev point to new point. Append fractional coords of new point to list."""
        self.canvas.create_oval(
            x - r,
            y - r,
            x + r,
            y + r,
            fill=self.fill_colour,
            width=0,
            tags="in_progress",
        )
        frac_points = self.current_label_frac_points
        if len(frac_points) > 0:
            x0, y0 = self._frac_to_canvas_coords(*frac_points[-1])
            self.canvas.create_line(
                x0, y0, x, y, fill=self.fill_colour, width=2.2, tags="in_progress"
            )
        self.current_label_frac_points.append((frac_x, frac_y))
        return None

    def _mouse_motion_poly(self, x: int, y: int) -> None:
        self.canvas.delete("animated")
        prev_point_frac_coords = self.current_label_frac_points[-1]
        x0, y0 = self._frac_to_canvas_coords(*prev_point_frac_coords)
        self.canvas.create_line(
            x0, y0, x, y, fill=self.fill_colour, width=2.2, tags="animated"
        )

    def finish_poly(self, event: tk.Event) -> None:
        """Submit current label to data_model, delete in progress gui stuff."""
        self.canvas.delete("in_progress")
        self.canvas.delete("animated")

        current_class = self.app.data_model.current_class
        label_type = self.app.data_model.labelling_type

        self.app.data_model.add_label(
            current_class,
            self.current_label_frac_points,
            label_type,
            self.app.brush_width,
        )
        self.app.treeview._update_treeview_from_labels(
            self.app.data_model.current_piece.labels
        )
        self.current_label_frac_points = []
        self.app.needs_updating = True

    """
    # SAM STUFF
    def _sam_suggest(self, canvas_x: int, canvas_y: int) -> None:
        self.canvas.delete("animated")
        arr_x, arr_y = self._canvas_to_arr_coords(canvas_x, canvas_y)
        polygon_arr_coords = self.app.data_model.sam_get_polygon_suggestion(
            arr_x, arr_y
        )
        self.sam_arr_poly = polygon_arr_coords
        canvas_coords = [
            self._arr_to_canvas_coords(x, y) for x, y in polygon_arr_coords
        ]
        self.canvas.create_polygon(
            *canvas_coords,
            outline=self.fill_colour,
            width=2.2,
            fill="",
            tags="animated",
        )

    def _increment_sam_hypothesis_idx(self, canvas_x: int, canvas_y: int) -> None:
        self.app.data_model.sam_hypothesis_index = (
            self.app.data_model.sam_hypothesis_index + 1
        ) % 3
        arr_x, arr_y = self._canvas_to_arr_coords(canvas_x, canvas_y)
        self._sam_suggest(arr_x, arr_y)

    def _place_sam_label(self, event: tk.Event):
        self.current_label_frac_points = [
            self._arr_to_frac_coords(arr_x, arr_y) for arr_x, arr_y in self.sam_arr_poly
        ]
        self.finish_poly(event)
    """
