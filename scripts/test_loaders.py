"""Smoke test — iterate one batch from each loader and print tensor shapes."""
import sys
import os

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import clip
import torch
from src.config_loader import load_config
from src.data_loaders.dataloader_factory import build_dataloaders


def main():
    config = load_config()
    loaders = build_dataloaders(config)

    for name, loader in loaders.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {name}")
        print(f"{'='*60}")
        print(f"  Total batches : {len(loader)}")
        print(f"  Total samples : {len(loader.dataset)}")

        # Grab one batch
        batch = next(iter(loader))

        if name == "clevr":
            images, counts = batch
            print(f"  Image batch   : {images.shape}")    # Expected: (32, 3, 224, 224)
            print(f"  Counts sample : {counts[:8].tolist()}")

            # Test tokenizing counting prompts
            count_range = config["prompts"]["counting"]["range"]
            template = config["prompts"]["counting"]["template"]
            prompts = [template.replace("{N}", str(n)) for n in count_range]
            tokens = clip.tokenize(prompts)
            print(f"  Prompt tokens : {tokens.shape}")    # Expected: (10, 77)

        elif name == "spatialsense":
            images, subjects, relations, objects, labels = batch
            print(f"  Image batch   : {images.shape}")    # Expected: (32, 3, 224, 224)
            print(f"  Sample triplet: '{subjects[0]}' {relations[0]} '{objects[0]}' (label={labels[0]})")

            # Test tokenizing a spatial prompt pair
            s, r, o = subjects[0], relations[0], objects[0]
            original = f"The {s} is {r} the {o}"
            swapped  = f"The {o} is {r} the {s}"
            tokens = clip.tokenize([original, swapped])
            print(f"  Prompt tokens : {tokens.shape}")    # Expected: (2, 77)

        elif name == "imagenet_v2":
            images, class_indices = batch
            print(f"  Image batch   : {images.shape}")    # Expected: (32, 3, 224, 224)
            print(f"  Class indices : {class_indices[:8].tolist()}")

    print(f"\n{'='*60}")
    print("All loaders passed shape checks!")


if __name__ == "__main__":
    main()
