import sys
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.append(".")
from src.tabulate import tabulate
from src.models import MaskedSequenceEncoder
from src.data import SequenceClassifcationDataset


def get_dataloaders(args):
    train_transforms = T.Compose(
        [
            T.Lambda(lambda x: torch.from_numpy(x).float()),
            T.RandomChoice(
                [
                    T.Lambda(lambda x: x + torch.randn_like(x) * args.aug_gamma),
                    T.Lambda(
                        lambda x: x
                        + (x / x.norm()) * torch.randn_like(x) * args.aug_gamma
                    ),
                ]
            ),
        ]
    )

    train_dataset = SequenceClassifcationDataset(
        args.data_dir,
        transform=train_transforms,
        split="train",
        source=args.source,
        pre_seq_len=args.pre_seq_len,
        suf_seq_len=args.suf_seq_len,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=True,
    )

    val_transforms = T.Compose(
        [
            T.Lambda(lambda x: torch.from_numpy(x).float()),
        ]
    )

    val_dataset = SequenceClassifcationDataset(
        args.data_dir,
        transform=val_transforms,
        split=args.split,
        source=args.source,
        pre_seq_len=args.pre_seq_len,
        suf_seq_len=args.suf_seq_len,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False,
    )

    return (
        train_dataloader,
        train_dataset.num_classes,
        val_dataloader,
        val_dataset.num_classes,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir")
    parser.add_argument("--source", default="features")
    parser.add_argument("--no-model", action="store_true")
    parser.add_argument("--pool", action="store_true")
    parser.add_argument("--split", default="val")
    parser.add_argument("--checkpoint-path", default="checkpoint.pt")

    parser.add_argument("--aug-gamma", type=float, default=0.01)
    parser.add_argument("--feat-dim", type=int, default=384)
    parser.add_argument("--hidden-dim", type=int, default=768)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-epochs", type=int, default=10)

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

        model.forward_model = model.forward
        model.forward = lambda x: model.forward_model(x)[:, 0, :]
    elif args.pool:
        if args.source == "features":
            model = nn.Identity()
            model.forward = (
                lambda x: (
                    x[:, : args.pre_seq_len, :].mean(dim=1)
                    + x[:, -args.suf_seq_len :, :].mean(dim=1)
                )
                / 2
            )

            task_dim = args.feat_dim
    else:
        if args.source == "features":
            model = nn.Identity()
            model.forward = lambda x: x[:, args.pre_seq_len, :]

            task_dim = args.feat_dim

    model.to(device)
    model.eval()

    (
        train_dataloader,
        num_train_classes,
        val_dataloader,
        num_val_classes,
    ) = get_dataloaders(args)

    task_head = nn.Linear(task_dim, num_train_classes)
    task_head.to(device)
    task_head.train()

    optimizer = torch.optim.AdamW(task_head.parameters(), lr=0.0001)

    print("Training linear task head...")

    train_accuracy, train_loss = 0.0, 0.0
    train_normalizer = 0.0

    for epoch in range(args.num_epochs):
        class_balance = torch.zeros(num_train_classes + 1).to(device)
        for sequence, y in tqdm(train_dataloader):
            optimizer.zero_grad()

            sequence = sequence[y != num_train_classes].to(device)
            y = y[y != num_train_classes].to(device)
            b, s, f = sequence.shape

            if b == 0:
                continue

            y = y.long()

            with torch.no_grad():
                x_enc = model(sequence)
            y_hat = task_head(x_enc)

            loss = F.cross_entropy(y_hat, y, ignore_index=num_train_classes)

            loss.backward()
            optimizer.step()

            train_accuracy += (y_hat.argmax(dim=-1) == y).float().mean().item()
            train_loss += loss.item()

            train_normalizer += 1.0

            class_balance += y.bincount(minlength=num_train_classes + 1)

    print("Class balance: ")
    print(100 * class_balance / class_balance.sum())

    task_head.eval()

    print("Evaluating linear classification...")

    val_top1_accuracy, val_top3_accuracy, val_loss = 0.0, 0.0, 0.0
    val_normalizer = 0.0

    for sequence, y in tqdm(val_dataloader):
        sequence = sequence[y != num_val_classes]
        y = y[y != num_val_classes]
        b, s, f = sequence.shape

        y = y.long()

        if b == 0:
            continue

        with torch.no_grad():
            x_enc = model(sequence)
            y_hat = task_head(x_enc)

        loss = F.cross_entropy(y_hat, y, ignore_index=num_val_classes)
        val_loss += loss.item()

        val_top1_accuracy += (y_hat.argmax(dim=-1) == y).float().mean().item()
        val_top3_accuracy += (
            (y_hat.topk(3, dim=-1).indices == y.unsqueeze(-1))
            .any(dim=-1)
            .float()
            .mean()
            .item()
        )

        val_normalizer += 1.0

    table = [
        [
            f"{100 * val_top1_accuracy / val_normalizer:.2f}%",
            f"{100 * val_top3_accuracy / val_normalizer:.2f}%",
            f"{val_loss / val_normalizer:.4f}",
            f"{100 * train_accuracy / train_normalizer:.2f}%",
            f"{train_loss / train_normalizer:.4f}",
        ]
    ]

    print(
        tabulate(
            table,
            headers=[
                "Val Top-1 Accuracy",
                "Val Top-3 Accuracy",
                "Val Loss",
                "Train Top-1 Accuracy",
                "Train Loss",
            ],
            tablefmt="fancy_grid",
        )
    )
