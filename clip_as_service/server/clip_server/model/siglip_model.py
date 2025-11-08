import torch
from clip_server.model.clip_model import CLIPModel
from clip_server.model.pretrained_models import _VISUAL_MODEL_IMAGE_SIZE
from transformers import AutoModel, AutoProcessor
# Originally from https://github.com/openai/CLIP. MIT License, Copyright (c) 2021 OpenAI

import io

import pillow_avif
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


class SigLIPModel(CLIPModel):
    def __init__(
        self,
        name: str,
        device: str = 'cpu',
        jit: bool = False,
        dtype: str = None,
        **kwargs
    ):
        super().__init__(name, **kwargs)
        self._name = name
        self._model = self.build_model(self._name, device)
        self._image_size = int(self._name.split('-')[-1])
    
    def build_model(self, ckpt: str, device: torch.device):
        # model = AutoModel.from_pretrained(ckpt)
        model = AutoModel.from_pretrained(ckpt, attn_implementation="flash_attention_2", torch_dtype=torch.float16)

        # Optimisations
        model = (
            model.to(device, memory_format=torch.channels_last)
            .eval()
        )
        # model = torch.compile(model, mode="reduce-overhead")
        model = torch.compile(
            model,
            mode="max-autotune",            # 2.3 起加入，触发 CUTLASS+Triton 搜索  [oai_citation:2‡GitHub](https://github.com/pytorch/pytorch/issues/96693?utm_source=chatgpt.com)
            # fullgraph=True,                 # 尽量不打 graph break
            # dynamic=False                   # SigLIP 输入(H,W)固定，关掉动态形状能多融合
        )
        return model

    @staticmethod
    def get_model_name(name: str):
        return self._name

    def encode_text(self, input_ids: 'torch.Tensor', attention_mask: 'torch.Tensor', **kwargs):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            text_features = self._model.get_text_features(
                input_ids=input_ids,
                # attention_mask=attention_mask
            )
        return text_features.to(torch.float32)

    def encode_image(self, pixel_values: 'torch.Tensor', **kwargs):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            image_features = self._model.get_image_features(pixel_values)
        return image_features.to(torch.float32)

    @property
    def model_name(self):
        return self.__class__.get_model_name(self._name)

    @property
    def image_size(self):
        return self._image_size


def _convert_image_to_rgb(image):
    return image.convert('RGB')


def _blob2image(blob):
    return Image.open(io.BytesIO(blob))


def _transform_blob(n_px):
    return Compose(
        [
            _blob2image,
            Resize((n_px, n_px), interpolation=BICUBIC),
            _convert_image_to_rgb,
            ToTensor(),
            Normalize(
                (0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5),
            ),
        ]
    )


def _transform_ndarray(n_px):
    return Compose(
        [
            ToTensor(),
            Resize(n_px, interpolation=BICUBIC),
            CenterCrop(n_px),
            Normalize(
                (0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5),
            ),
        ]
    )
