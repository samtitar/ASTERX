import sys
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from tqdm import tqdm
from tabulate import tabulate
from torch.utils.data import DataLoader

sys.path.append(".")
from src.engine import eval_one_epoch
from src.models import MaskedSequenceEncoder
from src.data import SequenceCandidateDataset, CandidateSampler


class ExperimentTracker:
    def __init__(self, num_candidates_steps):
        self.num_candidates_steps = num_candidates_steps

        self.step = 0
        self.top1_accuracies = []
        self.top5_accuracies = []
        self.top10_accuracies = []

    def log(self, dict):
        self.top1_accuracies.append(dict["val_top1_accuracy"])
        self.top5_accuracies.append(dict["val_top5_accuracy"])
        self.top10_accuracies.append(dict["val_top10_accuracy"])
        self.step += 1

    def print_results(self):
        table = [
            [
                self.num_candidates_steps[i],
                f"{self.top1_accuracies[i] * 100:.2f}%",
                f"{self.top5_accuracies[i] * 100:.2f}%",
                f"{self.top10_accuracies[i] * 100:.2f}%",
            ]
            for i in range(self.step)
        ]

        print(
            tabulate(
                table,
                headers=[
                    "Num Candidates",
                    "Top-1 Accuracy",
                    "Top-5 Accuracy",
                    "Top-10 Accuracy",
                ],
                tablefmt="fancy_grid",
            )
        )


def get_dataloader(num_candidates, args):
    transforms = T.Compose(
        [
            T.Lambda(lambda x: torch.from_numpy(x).float()),
        ]
    )

    candidate_sampler = CandidateSampler(
        args.data_dir,
        split=args.split,
        source=args.source,
        sampling_key="panel_level",
        transform=transforms,
        num_candidates=num_candidates - 1,
    )

    dataset = SequenceCandidateDataset(
        args.data_dir,
        candidate_sampler,
        split=args.split,
        source=args.source,
        transform=transforms,
        pre_seq_len=args.pre_seq_len,
        suf_seq_len=args.suf_seq_len,
        return_metadata=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False,
    )

    return dataloader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir")
    parser.add_argument("--source", default="features")
    parser.add_argument("--no-model", action="store_true")
    parser.add_argument("--split", default="val")
    parser.add_argument("--checkpoint-path", default="checkpoint.pt")

    parser.add_argument("--feat-dim", type=int, default=384)
    parser.add_argument("--hidden-dim", type=int, default=768)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--pre-seq-len", type=int, default=2)
    parser.add_argument("--suf-seq-len", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument(
        "--num-candidates-steps", nargs="+", type=int, default=[200, 500, 1000, 0]
    )
    args = parser.parse_args()

    args.track_failure_cases = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not args.no_model:
        model = MaskedSequenceEncoder(
            input_dim=args.feat_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            dropout=args.dropout,
        )

        model_head = nn.Linear(args.hidden_dim, args.feat_dim)

        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        model_head.load_state_dict(checkpoint["model_head_tok"])
    else:
        if args.source == "features":
            model = nn.Identity()
            model_head = nn.Identity()

            model.forward = (
                lambda x: (
                    x[:, : args.pre_seq_len, :].mean(dim=1)
                    + x[:, -args.suf_seq_len :, :].mean(dim=1)
                )
                / 2
            )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    experiment_tracker = ExperimentTracker(args.num_candidates_steps)
    best_acc = {"top1": 0, "top5": 0, "top10": 0}

    for num_candidates in args.num_candidates_steps:
        args.num_eval_candidates = num_candidates
        dataloader = get_dataloader(num_candidates, args)
        eval_one_epoch(
            model,
            model_head,
            dataloader,
            device,
            num_candidates,
            experiment_tracker,
            best_acc,
            args,
        )

        experiment_tracker.print_results()
