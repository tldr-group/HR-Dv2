"""Entry point for app."""

import tkinter as tk
from GUI import App, DataModel


if __name__ == "__main__":
    root = tk.Tk()
    root.title("SAMBA")

    data_model = DataModel()
    app = App(root, data_model)

    # app.load_image_from_filepaths(("data/bart_sem.tif",))
    app.treeview.n_classes = 4
    # app.load_labels_from_fp("img_0_labels.pkl")

    app.grid()
    root.mainloop()
