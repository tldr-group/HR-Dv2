"""Entry point for app."""

import tkinter as tk
from GUI import App, DataModel

# TODO:
# save classifier
# model choice in app / new classifier #
# random forest classifier
# abstract back into threaded process class

# Experiment:
# load a few examples
# add set of labels to N images
# train classifer with different models
# apply to whole dataset, measure mIoU as fn of model/n labelled examples

# add CRF? make classifier use predict proba and take argmax as in random forest

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
