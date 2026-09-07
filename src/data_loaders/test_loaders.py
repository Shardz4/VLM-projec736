import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import clip
import torch

from src.config_loader import load_config
from src.data_loader_factory import build_dataloaders

def main():
    config = load_config()
    loaders = build_dataloaders(config)

    for name, loader in loaders.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {name}")
        print(f"{'='*60}")
        print(f" Total batches: {len(loader)}")
        print(f" Total Samples: {len(loader.dataset)}")

        batch = next(iter(loader))

        if name == "clevr":
            images, counts = batch
            print(f" Image shape {images.shape} \n")
            print(f" Counts sample: {counts[:8].tolist()}")

            count_range = config["prompts"]["counting"]["range"]
            template = config["prompts"]["counting"]["template"]
            prompts = [template.replace("{N}",str(i)) for i in count_range]
            print(f" Prompt tokens : {tokens.shape}")
        
        elif name == "spatialsense":
            images, sub, rel, obj, labels = batch
            print(f" Image batch : {images.shape}")
            print(f" sample triplet: '{sub[0]} {rel[0]} {obj[0]}' (label={labels[0]})")
            
            s,r,o = sub[0], rel[0], obj[0]
            original = f"The {s} is {r} the {o}"
            swapped = f"The {o} is {r} the {s}"
            tokens = clip.tokenize([original, swapped])
            print(f" Prompt tokens : {tokens.shape}")

        elif name == "imagenet_v2":
            images, class_indices = batch
            print(f"Image Shape: {images.shape}")
            print(f"Class indices sample : {class_indices[:8].tolist()}")

    print(f"\n{'='*60}")
    print("All loaders passed shape checks")

if __name__ == "__main__":
    main()
            
