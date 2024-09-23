import os
import sys
import h5py
import json
import argparse
import numpy as np

import torch
from torch import nn
from torchvision import transforms as pth_transforms
from torchvision import models as torchvision_models

import utils
import src.vision_transformer as vits

from tqdm import tqdm
from data import HDF5ImageFolder


@torch.no_grad()
def extract_features(model, dataloader, device, data_path, split):
    o_file = h5py.File(f"{data_path}/{split}_features.hdf5", "w")

    result = {}
    for samples, comic_ids, panel_ids in tqdm(dataloader):
        samples = samples.to(device)
        features = model(samples).clone().cpu().numpy()

        for i, (comic_id, panel_id) in enumerate(zip(comic_ids, panel_ids)):
            if comic_id not in result:
                result[comic_id] = {}
            result[comic_id][panel_id] = features[i]

    indexing = {f"{split}_features": {}}
    for comic_id in result:
        comic_group = o_file.create_group(f"{split}_features/{comic_id}")
        comic_features = np.array(list(result[comic_id].values()))
        comic_group.create_dataset("feat_data", data=comic_features)

        indexing[f"{split}_features"][comic_id] = {}
        indexing[f"{split}_features"][comic_id]["feat_data"] = list(
            result[comic_id].keys()
        )

    with open(f"{data_path}/{split}_features_indexing.json", "w+") as f:
        json.dump(indexing, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("DINO feature extraction script")
    parser.add_argument(
        "--batch_size_per_gpu", default=128, type=int, help="Per-GPU batch-size"
    )
    parser.add_argument(
        "--pretrained_weights",
        default="",
        type=str,
        help="Path to pretrained weights to evaluate.",
    )
    parser.add_argument("--arch", default="vit_small", type=str, help="Architecture")
    parser.add_argument(
        "--patch_size", default=16, type=int, help="Patch resolution of the model."
    )
    parser.add_argument(
        "--checkpoint_key",
        default="teacher",
        type=str,
        help='Key to use in the checkpoint (example: "teacher")',
    )
    parser.add_argument(
        "--num_workers",
        default=10,
        type=int,
        help="Number of data loading workers per GPU.",
    )
    parser.add_argument("--data_path", default="/path/to/imagenet/", type=str)
    parser.add_argument("--split", default="train", type=str)
    args = parser.parse_args()

    transform = pth_transforms.Compose(
        [
            pth_transforms.Resize(256, interpolation=3),
            pth_transforms.CenterCrop(224),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    dataset = HDF5ImageFolder(
        args.data_path, split=args.split, transform=transform, return_metadata=True
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    if "vit" in args.arch:
        model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
        print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")
    elif "xcit" in args.arch:
        model = torch.hub.load("facebookresearch/xcit:main", args.arch, num_classes=0)
    elif args.arch in torchvision_models.__dict__.keys():
        model = torchvision_models.__dict__[args.arch](num_classes=0)
        model.fc = nn.Identity()
    else:
        print(f"Architecture {args.arch} non supported")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    utils.load_pretrained_weights(
        model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size
    )

    model.eval()

    extract_features(model, data_loader, device, args.data_path, args.split)
