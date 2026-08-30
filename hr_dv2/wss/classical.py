import torch
from torchvision.transforms import ToTensor  # type: ignore
from kornia.filters import (
    bilateral_blur,
    gaussian_blur2d,
    max_blur_pool2d,
    median_blur,
    sobel,
    laplacian,
)
from kornia.feature import dog_response_single, hessian_response
from PIL import Image
import numpy as np


from os import getcwd
from os.path import join
from dataclasses import dataclass


@dataclass
class Features:
    min_sigma: float = 0
    max_sigma: float = 16
    use_gauss: bool = True
    use_sobel: bool = True
    use_hess: bool = True
    use_dog: bool = True
    use_bilateral: bool = False  # bilateral is pretty slow
    use_median: bool = True
    use_lap: bool = True


DEFAULT_FEATURES = Features()


def classical_featurise(
    x: torch.Tensor, feature_set: Features, to_half: bool = True
) -> torch.Tensor:
    """Perform GPU accelerated classical featurisation of image x

    :param x: (B, C, H, W) image tensor in range [0.0, 1.0] and dtype torch.float32
    :type x: torch.Tensor
    :param feature_set: dataclass of features you want and lengthscales you want them over
    :type feature_set: Features
    :return: (B, N_FEATS, H, W) tensor of classical features
    :rtype: torch.Tensor
    """
    # bounds check
    assert torch.max(x) <= 1.0
    assert torch.min(x) >= 0.0

    feat_list: list[torch.Tensor] = []
    gauss_list: list[torch.Tensor] = []

    sigma_min = feature_set.min_sigma
    if sigma_min == -1:
        sigma_min = 0.5
    elif sigma_min == 0.0:
        # weka-style zero-sigma: appending image and 0-sigma hess & sobel
        zeroscale_hess = hessian_response(x)
        zeroscale_sobel = sobel(x)
        feat_list = [x, zeroscale_sobel, zeroscale_hess]
        sigma_min = 1.0

    b2_min, b2_max = np.log2(sigma_min), np.log2(feature_set.max_sigma)
    num_sigma = int(b2_max - b2_min) + 1
    sigmas = [(2**i) * sigma_min for i in range(num_sigma)]

    def _k_from_s(s: float) -> tuple[int, int]:
        if s == 1.0:
            return (1, 1)
        else:
            return (int(s) + 1, int(s) + 1)

    for sigma in sigmas:
        gk = _k_from_s(4 * sigma)
        k = _k_from_s(sigma)
        y_gauss = gaussian_blur2d(x, gk, (sigma, sigma))
        gauss_list.append(y_gauss)
        if feature_set.use_gauss:
            feat_list.append(y_gauss)
        if feature_set.use_sobel:
            y_sobel = sobel(y_gauss)
            feat_list.append(y_sobel)
        if feature_set.use_hess:
            y_hess = hessian_response(y_gauss)
            feat_list.append(y_hess)
        if feature_set.use_median:
            y_median = median_blur(x, k)
            feat_list.append(y_median)
        if feature_set.use_lap:
            y_lap = laplacian(y_gauss, k)
            feat_list.append(y_lap)

    if feature_set.use_bilateral:
        for spatial_radius in [5, 11]:
            k = _k_from_s(3 * spatial_radius)
            for color_sigma in [50.0 / 255, 100.0 / 255]:
                y_bilateral = bilateral_blur(
                    x, k, color_sigma, (spatial_radius, spatial_radius)
                )
                feat_list.append(y_bilateral)
    if feature_set.use_dog:
        for i, sigma_1 in enumerate(gauss_list):
            for j in range(i):
                sigma_2 = gauss_list[j]
                y_dog = sigma_2 - sigma_1
                feat_list.append(y_dog)
    rgb_feats = torch.cat(feat_list)
    n_c, n_channels, h, w = rgb_feats.shape
    feats_3D = rgb_feats.view(n_c * n_channels, h, w)
    if to_half:
        feats_3D = feats_3D.to(torch.float16)
    feats_3D = torch.nan_to_num(feats_3D, 0)
    return feats_3D.unsqueeze(0)


if __name__ == "__main__":
    data_path = join(getcwd(), "upsampling_vit_experiments", "tests", "data")
    img = Image.open(f"{data_path}/chair.jpg")
    tr = ToTensor()
    x = tr(img)
    x = x.unsqueeze(0)
    x = x.to(torch.float32)
    print(x.shape)
    f = classical_featurise(x, DEFAULT_FEATURES)
    print(f[0, 0, :10, :10])
    print(torch.min(f), torch.max(f))
    print(f.shape)
