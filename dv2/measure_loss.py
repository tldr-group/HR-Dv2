import torch
from PIL import Image
from torchvision.transforms import ToTensor, CenterCrop


from dinov2.dinov2.configs import load_config
from dinov2.dinov2 import 
from dinov2.dinov2.train import SSLMetaArch

cfg = load_config("ssl_default_config")

to_tensor = ToTensor()
crop = CenterCrop(224)

# dinov2_vits14 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
# dinov2_vits14.eval()
# dinov2_vits14.cuda()

meta_arch = SSLMetaArch(cfg)
# meta_arch.eval()
meta_arch.cuda()


test_img = Image.open("data/BSD300/train/3096.jpg")
test_tensor = crop(to_tensor(test_img)).unsqueeze(0)
test_tensor = test_tensor.to("cuda")

out = meta_arch.forward_backward(test_tensor, 0.07)

# out = dinov2_vits14.forward_features(test_tensor)
print(cfg)
