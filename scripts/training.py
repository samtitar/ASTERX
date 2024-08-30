import sys
import wandb
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.append(".")

from src.models import MaskedSequenceEncoder
from src.engine import train_one_epoch, eval_one_epoch
from src.data import SequenceCandidateDataset, CandidateSampler


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

    train_candidate_sampler = CandidateSampler(
        args.data_dir,
        split="train",
        source=args.source,
        sampling_key=args.sampling_key,
        sampling_strategy=args.sampling_strategy,
        negate_sampling=args.negate_sampling,
        transform=train_transforms,
        num_candidates=args.num_train_candidates - 1,
    )

    train_dataset = SequenceCandidateDataset(
        args.data_dir,
        train_candidate_sampler,
        split="train",
        source=args.source,
        transform=train_transforms,
        pre_seq_len=args.pre_seq_len,
        suf_seq_len=args.suf_seq_len,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=True,
    )

    val_transforms = T.Compose(
        [
            T.Lambda(lambda x: torch.from_numpy(x).float()),
        ]
    )

    val_candidate_sampler = CandidateSampler(
        args.data_dir,
        split="val",
        source=args.source,
        sampling_key="panel_level",
        transform=val_transforms,
        num_candidates=args.num_eval_candidates - 1,
    )

    val_dataset = SequenceCandidateDataset(
        args.data_dir,
        val_candidate_sampler,
        split="val",
        source=args.source,
        transform=val_transforms,
        pre_seq_len=args.pre_seq_len,
        suf_seq_len=args.suf_seq_len,
        return_metadata=True,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False,
    )

    if args.num_eval_candidates < 1:
        args.num_eval_candidates = len(val_dataset)

    return train_dataloader, val_dataloader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir")
    parser.add_argument("--source", default="features")

    parser.add_argument("--feat-dim", type=int, default=384)
    parser.add_argument("--hidden-dim", type=int, default=768)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--aug-gamma", type=float, default=0.001)
    parser.add_argument("--sampling-key", default="panel_level")
    parser.add_argument("--sampling-strategy", default="random")
    parser.add_argument("--negate-sampling", action="store_true")

    parser.add_argument("--pre-seq-len", type=int, default=2)
    parser.add_argument("--suf-seq-len", type=int, default=2)
    parser.add_argument("--num-train-candidates", type=int, default=500)
    parser.add_argument("--num-eval-candidates", type=int, default=5000)
    parser.add_argument("--no-cls-optimization", action="store_true")
    parser.add_argument("--no-tok-optimization", action="store_true")

    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=1)

    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.00001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-path", default="checkpoint.pt")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    wandb.init(project="sequence-encoding")
    wandb.config.update(vars(args))

    args.output_path = args.output_path.replace("[wandb_id]", wandb.run.id)
    args.track_failure_cases = False

    assert not (args.no_cls_optimization and args.no_tok_optimization), (
        "Both optimization tasks are disabled. " "This is not a valid configuration."
    )

    train_dataloader, val_dataloader = get_dataloaders(args)

    model = MaskedSequenceEncoder(
        input_dim=args.feat_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
    )

    model_head = {
        "cls": nn.Linear(args.hidden_dim, 1),
        "tok": nn.Linear(args.hidden_dim, args.feat_dim),
    }

    optimizer = torch.optim.AdamW(
        list(model.parameters())
        + list(model_head["cls"].parameters())
        + list(model_head["tok"].parameters()),
        lr=args.lr,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    best_acc = {"top1": 0, "top5": 0, "top10": 0}
    early_stop = 0

    for epoch in range(args.epochs):
        train_one_epoch(
            model, model_head, train_dataloader, device, epoch, optimizer, wandb, args
        )

        new_best_acc = eval_one_epoch(
            model,
            model_head["tok"],
            val_dataloader,
            device,
            epoch,
            wandb,
            best_acc,
            args,
        )

        if new_best_acc["top5"] > best_acc["top5"]:
            best_acc = new_best_acc
            early_stop = 0

            state_dict = {
                "model": model.state_dict(),
                "model_head_tok": model_head["tok"].state_dict(),
                "model_head_cls": model_head["cls"].state_dict(),
            }

            torch.save(state_dict, args.output_path)
        else:
            early_stop += 1

            if early_stop > 10:
                break
