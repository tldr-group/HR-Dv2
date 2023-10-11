# ViT Testing

Repo for testing various [Vision Transformer (ViT)](https://arxiv.org/abs/2010.11929) feature learning architectures and their suitability for training/finetuning on a (hypothetical) large micrograph dataset. [Masked Autoencoders (MAES)](https://arxiv.org/abs/2111.06377) are promising architectures for self-supervised feature learning on large scale databases and there exist a variety of models pretrained on very large image datasets. These pretrained feature learners can then be applied to downstream tasks, either by fine-tuning or freezing the encoder backbone and training a task-specific head.  

## Models considered:

- **Vanilla Masked Autoencoder**: backbone for most of the following models. Images are patched into tokens, up to 75% of these patches are masked and the input is fed into a transformer encdoer/decoder network. The task is to learn to reconstruct the whole image given the unmasked patches. [[`Paper`](https://arxiv.org/abs/2111.06377)]
- **Hiera**: a hierarchical Vision Transformer where early layers have more spatial resolution and later layers have more features. Simplifies previous hierarchical architectures (hierarchitectures?) with clever strides. [[`Code`](https://github.com/facebookresearch/hiera)] [[`Paper`](https://arxiv.org/abs/2306.00989)]
- **DINOv2**: student-teacher model for self-supervised feature learning with ViT backbone. Training DS was 1.2B images - they claim it's been trained on a wide enough corpus that fine-tuning is unnecessary and show good performance by training linear heads on top of the frozen backbone. It's a more complicated architecture than the other models. [[`Code`](https://github.com/facebookresearch/dinov2/tree/main)] [[`Paper`](https://arxiv.org/abs/2304.07193)]

## Assessing Performance:

The models will be run in inference mode on 3 different small datasets: [BSD300](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/), [ImageNetSketch](https://github.com/HaohanWang/ImageNet-Sketch/tree/master) and some micrographs - these are all subsets that are roughly 100 images big. The idea is that BSD300 represents in-distribution images of real-world scenes, ImageNetSketch represents somewhat out-of-distribtion images and the micrographs are 'very-out-of-distribution'. The average loss (the scales and type of loss will differ over different models) over the datasets will be measured and compared. My hypothesis is that the loss will be much higher for the micrographs than for the BSD300 and INS, because they are quite different in terms of semantics (i.e all areas equally important, no specific subject, lots of repitition).

## Finetuning Suitability:

| Model | Params | Checkpoint | Training Scripts | Resolution |
| ----- | ------ | ---------- | ---------------- | ---------- |
| ViT |   | Full | Y | ? |
| Hiera |   | Full | N | 224x224 |
| DINOv2 | 21M/86M/300M | Partial | N | 224/516 |

## DINOv2 upscaling:

DINOv2 checkpoints are trained with a patch size of 14x14, meaning the features produced are at that resolution. It can be trained from scratch (costly, long) with different patch sizes. I propose a simpler solution: pixel shift the image by 7, 4 and 2 pixels in the 4/8 directions in a Von Neumann/Moore neighbourhood with periodic boundaries, compute the features for each shift then average.  