import torch
from torch import Tensor
from typing import List
from torchvision.io import decode_jpeg, decode_image   # 0.20+ 已支持 GPU nvJPEG
from torchvision.transforms.functional import resize, convert_image_dtype
import torchvision
import io
from PIL import Image


class PreprocessBlobsToGpu:
    def __init__(self, n_px: int = 384, dtype: torch.dtype = torch.float32, device: torch.device | str = "cuda"):
        self.n_px = n_px
        self.dtype = dtype
        self.device = device
        self._MEAN = torch.tensor([0.5, 0.5, 0.5], device=self.device).view(1, 3, 1, 1)
        self._STD  = torch.tensor([0.5, 0.5, 0.5], device=self.device).view(1, 3, 1, 1)

    def __call__(self, jpeg_blobs: List[bytes]) -> Tensor:
        # ---------- 1. JPEG → Tensor (GPU) ----------
        # nvJPEG 一次性批量解码；decode_image 输出 CxHxW, uint8, device='cuda'
        imgs = [torch.frombuffer(b, dtype=torch.uint8) for b in jpeg_blobs]  # 每张图 CxHxW in uint8
        try:
            imgs = decode_jpeg(imgs, mode=torchvision.io.ImageReadMode.RGB, device=self.device)
        except Exception as e:
            imgs = [decode_image(img, mode=torchvision.io.ImageReadMode.RGB) for img in imgs]

        imgs = torch.stack(imgs).to(self.device)  # → (B,3,H,W)

        # ---------- 2. Resize ----------
        # torchvision 0.20 的 resize/center_crop 均支持 GPU Kernel
        imgs = resize(imgs, [self.n_px, self.n_px], antialias=True)

        # ---------- 3. uint8 → float16/32 ∈ [0,1] ----------
        imgs = convert_image_dtype(imgs, dtype=self.dtype)  # /255 转 float 并保持 device

        # ---------- 4. Normalize to [-1,1] ----------
        imgs = (imgs - self._MEAN).div_(self._STD)               # in-place，节省显存

        return imgs

def test_preprocess_blobs_to_gpu():
    img = Image.open('tests/both_motion_blur.png')

    img_bytes = io.BytesIO()
    img.save(img_bytes, format="JPEG")
    t = PreprocessBlobsToGpu(n_px=384)
    pixel_values = t([img_bytes.getvalue()])

    print(pixel_values.shape)
    im_arr = ((pixel_values * 0.5 + 0.5) * 255).to(torch.uint8).squeeze(0).permute(1, 2, 0).cpu().numpy()
    Image.fromarray(im_arr)