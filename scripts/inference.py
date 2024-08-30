import sys
import h5py
import json
import argparse
import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms as T

from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.append(".")
from models import MaskedSequenceEncoder
from data import SequenceClassifcationDataset


def get_dataloaders(args):
    transforms = T.Compose(
        [
            T.Lambda(lambda x: torch.from_numpy(x).float()),
        ]
    )

    train_dataset = SequenceClassifcationDataset(
        args.data_dir,
        transform=transforms,
        split="train",
        source=args.source,
        pre_seq_len=args.pre_seq_len,
        suf_seq_len=args.suf_seq_len,
        return_metadata=True,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False,
    )

    val_dataset = SequenceClassifcationDataset(
        args.data_dir,
        transform=transforms,
        split="val",
        source=args.source,
        pre_seq_len=args.pre_seq_len,
        suf_seq_len=args.suf_seq_len,
        return_metadata=True,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False,
    )

    test_dataset = SequenceClassifcationDataset(
        args.data_dir,
        transform=transforms,
        split="test",
        source=args.source,
        pre_seq_len=args.pre_seq_len,
        suf_seq_len=args.suf_seq_len,
        return_metadata=True,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False,
    )

    return (
        train_dataloader,
        val_dataloader,
        test_dataloader,
    )


@torch.no_grad()
def extract_features(model, dataloader, device, data_path, split):
    o_file = h5py.File(f"{data_path}/{split}_sequence_features.hdf5", "w")

    result = {}
    for samples, _, comic_ids, panel_ids in tqdm(dataloader):
        samples = samples.to(device)
        features = model(samples).clone().cpu().numpy()

        for i, (comic_id, panel_id) in enumerate(zip(comic_ids, panel_ids)):
            if comic_id not in result:
                result[comic_id] = {}
            result[comic_id][panel_id] = features[i]

    indexing = {f"{split}_sequence_features": {}}
    for comic_id in result:
        comic_group = o_file.create_group(f"{split}_sequence_features/{comic_id}")
        comic_features = np.array(list(result[comic_id].values()))
        comic_group.create_dataset("feat_data", data=comic_features)

        indexing[f"{split}_sequence_features"][comic_id] = {}
        indexing[f"{split}_sequence_features"][comic_id]["feat_data"] = list(
            result[comic_id].keys()
        )

    with open(f"{data_path}/{split}_sequence_features_indexing.json", "w+") as f:
        json.dump(indexing, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir")
    parser.add_argument("--source", default="features")
    parser.add_argument("--no-model", action="store_true")
    parser.add_argument("--split", default="val")
    parser.add_argument("--checkpoint-path", default="checkpoint.pt")

    parser.add_argument("--aug-gamma", type=float, default=0.01)
    parser.add_argument("--feat-dim", type=int, default=384)
    parser.add_argument("--hidden-dim", type=int, default=768)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--pre-seq-len", type=int, default=2)
    parser.add_argument("--suf-seq-len", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not args.no_model:
        model = MaskedSequenceEncoder(
            input_dim=args.feat_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            dropout=args.dropout,
        )

        task_dim = args.hidden_dim

        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
    else:
        if args.source == "features":
            model = nn.Identity()
            # model.forward = (
            #     lambda x: (
            #         x[:, : args.pre_seq_len, :].mean(dim=1)
            #         + x[:, -args.suf_seq_len :, :].mean(dim=1)
            #     )
            #     / 2
            # )

            model.forward = lambda x: x[:, args.pre_seq_len, :]

            task_dim = args.feat_dim

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    (
        train_dataloader,
        val_dataloader,
        test_dataloader,
    ) = get_dataloaders(args)

    extract_features(model, train_dataloader, device, args.data_dir, "train")
    extract_features(model, val_dataloader, device, args.data_dir, "val")
    extract_features(model, test_dataloader, device, args.data_dir, "test")
