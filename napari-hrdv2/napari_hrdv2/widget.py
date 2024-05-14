from qtpy.QtWidgets import QWidget, QPushButton, QVBoxLayout
from napari import Viewer  # type: ignore
from napari.layers import Layer, Image
from napari.utils.events import Event

import numpy as np

# import torch

# use sklearn lasso for classifier?


class SegWidget(QWidget):
    def __init__(self, napari_viewer: Viewer) -> None:
        super().__init__()
        self.viewer = napari_viewer

        self.image_layers: list[Image] = self._get_image_layers(self.viewer)
        self.image_features: list[np.ndarray] = []

        main_layout = QVBoxLayout()

        self.test = QPushButton("Test")
        self.test.clicked.connect(self._on_test_pressed)
        self.test.setEnabled(True)
        main_layout.addWidget(self.test)

        self.setLayout(main_layout)

    def _get_image_layers(self, viewer: Viewer) -> list[Image]:
        out = []
        for layer in viewer.layers:
            if type(layer) == Image:
                out.append(layer)
        return out
    
    def _featurise(self,) -> None:
        # loop through all open image layers
        # transform data
        # feed into net
        # store output in self.image_features

    def _on_test_pressed(self, event: Event) -> None:
        self._get_layers(self.viewer)
