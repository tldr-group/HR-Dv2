# High-Res DINOv2 (HR-Dv2)

Upsampling spatialised features from vision transformer (ViT) models like DINO and DINOv2 for unsupervised and weakly-supervised materials segmentation.

<p align="center">
    <img src="repo/gh_header.png">
</p>
Image: Shared PCA visualisations of the high(er)-resolution features.

See `minimum_example.ipynb` for usage.
GPU recommended - if running into memory errors try using 'forward_sequential' mode. 



## Weakly-supervised segmentation

The folder `app/` contains a very bare-bones weakly supervised segmentation app for testing your own data. Run (with .venv or conda env activated) from the root folder":
```
python app/main.py
```
Note the app resize longest side to 518 by default, so may take a while to compute features on low-end devices, or may downsample large images.

Tkinter theme from user [rdbende on github](https://github.com/rdbende/Azure-ttk-theme)

<p align="center">
    <img src="repo/wss_workflow_cell.png">
</p>


<p align="center">
    <img src="repo/wss_supp_examples.png">
</p>


## Install:

Requires Python 3.10 or greater.

```
cd HR-Dv2/
git clone https://github.com/facebookresearch/dinov2
pip install -r dinov2/requirements.txt -r dinov2/requirements-dev.txt
pip install -e .
```

To install with conda (recommended, gets pytorch sorted):
```
cd HR-Dv2/
git clone https://github.com/facebookresearch/dinov2
conda env create -f dinov2/conda.yaml
conda activate dinov2
pip install -e .
```

To compare with [FeatUp](https://github.com/mhamilton723/FeatUp), a different feature upsampler:
```
pip install git+https://github.com/mhamilton723/FeatUp
```

### Zip (for submission):

```
zip -r hrdv2.zip . -x@zipignore.txt
```
