from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
from PIL import Image
import clip

class CLIPEncoder:

    def __init__(self, model_name: str = "ViT-B/32", device: str = "cuda"):
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)
        self.model_name = model_name

        self.model, self.clip_preprocess = clip.load(model_name, device=device)
        self.model.eval()
    
    @property
    def embedding_dim(self):
        return self.model.visual.output_dim

    @property
    def logit_scale(self) -> torch.Tensor:
        return self.model.logit_scale.exp()
    
    @torch.no_grad()
    def encode_images(self, images: Union[torch.Tensor, List[Image.Image]]) -> torch.Tensor:
        if isinstance(images, list):
            images = torch.stack([self.preprocess(img) for img in images])
        
        images = images.to(self.device)
        images_features = self.model.encode_image(images)
        image_features = image_features . image_features.norm(dim=-1, keepdim=True)
        return image_features
    
    @torch.no_grad()
    def encode_texts(self, text_list: List[str]) -> torch.Tensor:
        tokens = clip.tokenize(text_list).to(self.device)
        text_features = self.model.encode_text(tokens)
        text_features = text_features.float()

        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features
    
    @torch.no_grad()
    def compute_similarity(self, image_features: torch.Tensor, text_features: torch.Tensor, use_logit_scale: bool = True) -> torch.Tensor:
        similarity = image_features @ text_features.T
        if use_logit_scale:
            logit_scale = self.model.logit_scale.exp().float()
            return logit_scale * similarity
        return similarity
    
    @torch.no_grad()
    def zero_shot_classify(
        self,
        images: torch.Tensor,
        prompt_templates: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        img_feats = self.encode_images(images)
        txt_feats = self.encode_texts(prompt_templates)
        logits = self.compute_similarity(img_feats, txt_feats, use_logit_scale=True)
        probs = logits.softmax(dim=-1)
        preds = probs.argmax(dim=-1)
        return probs, preds
    
    def _repr_(self) -> str:
        return (
            f"CLIPEncoder(model='{self.model_name}',"
            f"device='{self.device}',"
            f"embed_dim={self.embedding_dim},"
            f"logit_scale={self.logit_scale.item():.2f})"
        )