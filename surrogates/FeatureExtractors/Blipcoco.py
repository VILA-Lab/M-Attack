import torch
from transformers import (
    Blip2VisionModel,
    Blip2VisionConfig,
    Blip2Processor,
    Blip2Model,
    BlipImageProcessor,
)
from torchvision import transforms
from .Base import BaseFeatureExtractor
device = "cuda" if torch.cuda.is_available() else "cpu"
OPENAI_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]
IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]


class BlipcocoFeatureExtractor(BaseFeatureExtractor):
    def __init__(self):
        super(BlipcocoFeatureExtractor, self).__init__()
        self.normalizer = transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.Lambda(lambda img: torch.clamp(img, 0.0, 255.0) / 255.0),
            transforms.CenterCrop(224),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)), # CLIP imgs mean and std.
        ]
    )
        self.processor = BlipImageProcessor.from_pretrained("Salesforce/blip-itm-base-coco")
        self.model = Blip2Model.from_pretrained("Salesforce/blip-itm-base-coco")
        self.device = torch.device("cuda")
        self.eval().requires_grad_(False)

    def forward(self, x):
        inputs = dict(pixel_values=self.normalizer(x))
        inputs["pixel_values"] = inputs["pixel_values"].to(device)
        outputs = self.model.get_image_features(**inputs)
        pooler_output = outputs.pooler_output
        image_features = pooler_output / pooler_output.norm(dim=1, keepdim=True)
        return image_features
