import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from typing import Callable, Any, Literal, List
#from idlelib.tooltip import Hovertip


class HighlightButton(tk.Frame):
    """Button inside frame who's background colour can be changed to act as a highlight."""

    prev_btn: tk.Frame | None = None
    highlight_colour: str = "#1f77b4"

    def __init__(self, parent: tk.Frame) -> None:
        """Init frame and button. This is a tk.Frame (NOT ttk) as you can't set ttk colours w/out style nonsense."""
        super().__init__(parent)

    def _get_photoimage(self, image_path: str) -> ImageTk.PhotoImage:
        # it's own fn so we can save a reference to it so it doesn't get gc'd
        image = Image.open(f"{image_path}")
        resized = image.resize((16, 16))
        photoimage = ImageTk.PhotoImage(resized)
        return photoimage

    def _init_button(
        self,
        photoimage: ImageTk.PhotoImage,
        command: Callable,
        tooltip_txt: str,
    ) -> None:
        btn = ttk.Button(self, image=photoimage, width=1, command=lambda: self._toggle_btn_and_call_fn(self, command))  # type: ignore
        btn.grid(row=0, column=0, padx=(3, 3), pady=(3, 3))
        #btn_tooltip = Hovertip(self, text=tooltip_txt)

    def _toggle_btn_and_call_fn(
        self,
        frame_btn: tk.Frame,
        function: Callable,
    ) -> Any:
        # prev_btn is class variable shared between all these toggle buttons which is reference to last clicked button. If it is set then reset its stlye
        if HighlightButton.prev_btn is not None:
            HighlightButton.prev_btn["bg"] = frame_btn["bg"]
            HighlightButton.prev_btn["relief"] = frame_btn["relief"]
        frame_btn["bg"] = HighlightButton.highlight_colour  # the theme colour
        frame_btn["relief"] = "sunken"
        HighlightButton.prev_btn = frame_btn
        function()

    def _change_prev_btn_colour(self, colour: str) -> None:
        HighlightButton.highlight_colour = colour
        if HighlightButton.prev_btn is not None:
            HighlightButton.prev_btn["bg"] = HighlightButton.highlight_colour
            HighlightButton.prev_btn["relief"] = "sunken"


class TextMenuFrame(ttk.Frame):
    """Frame holding a text widget and a menu widget (slider or dropdown)."""

    def __init__(
        self,
        parent: tk.Frame | ttk.LabelFrame,
        fn: Callable,
        text: str,
        widget_type: Literal["slider", "dropdown"] = "slider",
        widget_params: List = [0.0, 1.0],
    ) -> None:
        """Init and grid text and slider."""
        super().__init__(parent)
        txt = ttk.Label(self, text=text)
        txt.grid(row=0, column=0, sticky="W", padx=(5, 5))
        self.menu: tk.Widget
        if widget_type == "slider":
            from_, to = widget_params
            self.menu = ttk.Scale(self, command=fn, from_=from_, to=to)
        elif widget_type == "dropdown":
            temp_var = tk.IntVar()
            self.menu = ttk.OptionMenu(
                self, temp_var, widget_params[0], *widget_params, command=fn
            )
        self.menu.grid(row=0, column=1, padx=(5, 5))
