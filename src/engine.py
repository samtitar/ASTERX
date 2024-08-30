import time
import torch
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm

fmi = lambda x: x.float().mean().item()


def train_one_epoch(
    model, model_head, dataloader, device, epoch, optimizer, logger, args
):
    print(f"Training epoch {epoch}")

    model.to(device)
    model.train()

    model_head["cls"].to(device)
    model_head["cls"].train()

    model_head["tok"].to(device)
    model_head["tok"].train()

    running_loss, running_cls_accuracy, running_tok_accuracy = 0.0, 0.0, 0.0
    step = 0

    for sequence1, sequence2, sequence3, candidates, y in tqdm(dataloader):
        b, s, f = sequence1.shape
        optimizer.zero_grad()

        sequence = sequence2.to(device)

        secondary_sequence = sequence1.to(device)
        switch_sequence = (torch.rand(b) < 0.5).to(device)
        secondary_sequence[switch_sequence] = sequence3.to(device)[switch_sequence]

        candidates = candidates.to(device)
        y = y.to(device)

        mask = torch.zeros((b, s), dtype=torch.bool, device=device)
        mask[:, args.pre_seq_len] = True

        x_enc = model(sequence, secondary_sequence=secondary_sequence, mask=mask)

        # CLS token representation to binary classification
        y_hat_cls = model_head["cls"](x_enc[:, 0]).squeeze(1).sigmoid()

        # Masked-token specific representation to classification
        num_candidates = candidates.shape[1]
        tok_enc = model_head["tok"](x_enc[:, args.pre_seq_len + 1])
        # y_hat_tok = F.cosine_similarity(tok_enc.unsqueeze(1), candidates, dim=-1)
        y_hat_tok = -F.pairwise_distance(tok_enc.unsqueeze(1), candidates, p=2)
        y_hat_tok[candidates.sum(dim=-1) == 0] = -1

        loss = 0.0
        if not args.no_cls_optimization:
            loss += F.binary_cross_entropy(y_hat_cls, switch_sequence.float())
        if not args.no_tok_optimization:
            loss += F.cross_entropy(y_hat_tok, y)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_cls_accuracy += fmi((y_hat_cls > 0.5) == switch_sequence)
        running_tok_accuracy += fmi(y_hat_tok.argmax(dim=-1) == y)
        step += 1

        if step % 25 == 24:
            logger.log(
                {
                    "train_loss": running_loss / 24,
                    "train_cls_accuracy": running_cls_accuracy / 24,
                    "train_tok_accuracy": running_cls_accuracy / 24,
                    "train_epoch": epoch,
                }
            )

            running_loss, running_cls_accuracy, running_tok_accuracy = 0.0, 0.0, 0.0


def eval_one_epoch(
    model, model_head, dataloader, device, epoch, logger, best_acc, args
):
    print(f"Evaluation epoch {epoch}")

    best_acc = best_acc.copy()

    model.to(device)
    model.eval()

    model_head.to(device)
    model_head.eval()

    loss, top1_accuracy, top5_accuracy, top10_accuracy = 0.0, 0.0, 0.0, 0.0
    failure_cases = []

    total_similarity, similarity_samples = 0.0, 0.0

    for _, sequence, _, candidates, y, (s_comic, s_panel), (c_comics, c_panels) in tqdm(
        dataloader
    ):
        b, s, f = sequence.shape

        sequence = sequence.to(device)
        candidates = candidates.to(device)
        y = y.to(device)

        c_comics = np.array(c_comics)
        c_panels = np.array(c_panels)

        with torch.no_grad():
            # x_enc = model(sequence)[:, args.pre_seq_len + 1]
            x_enc = model(sequence)
            x_enc = model_head(x_enc)

            num_candidates = candidates.shape[1]

            # y_hat = F.cosine_similarity(x_enc.unsqueeze(1), candidates, dim=-1)
            y_hat = -F.pairwise_distance(x_enc.unsqueeze(1), candidates, p=2)
            y_hat[candidates.sum(dim=-1) == 0] = -1

        loss += F.cross_entropy(y_hat, y).item()
        
        top1_accuracy += fmi(
            (y_hat.topk(1, dim=-1).indices == y.unsqueeze(-1)).any(dim=-1)
        )

        top5_accuracy += fmi(
            (y_hat.topk(5, dim=-1).indices == y.unsqueeze(-1)).any(dim=-1)
        )

        top10_accuracy += fmi(
            (y_hat.topk(10, dim=-1).indices == y.unsqueeze(-1)).any(dim=-1)
        )

        for i in range(b):
            if y_hat[i].argmax() == y[i]:
                total_similarity += y_hat[i].max().item()
                similarity_samples += 1

        # Store samples where the model failed in terms of top-1 accuracy
        if args.track_failure_cases:
            for i in range(b):
                if y_hat[i].argmax() != y[i]:
                    failure_cases.append(
                        {
                            "x_enc": x_enc[i],
                            "sequence": sequence[i],
                            "candidates": candidates[i],
                            "s_comic": s_comic[i],
                            "s_panel": s_panel[i],
                            "c_comics": c_comics[:, i],
                            "c_panels": c_panels[:, i],
                            "y": y[i],
                            "y_hat": y_hat[i],
                        }
                    )

    best_acc["top1"] = max(best_acc["top1"], top1_accuracy / len(dataloader))
    best_acc["top5"] = max(best_acc["top5"], top5_accuracy / len(dataloader))
    best_acc["top10"] = max(best_acc["top10"], top10_accuracy / len(dataloader))

    logger.log(
        {
            "val_loss": loss / len(dataloader),
            "val_top1_accuracy": top1_accuracy / len(dataloader),
            "val_top5_accuracy": top5_accuracy / len(dataloader),
            "val_top10_accuracy": top10_accuracy / len(dataloader),
            "val_max_top1_accuracy": best_acc["top1"],
            "val_max_top5_accuracy": best_acc["top5"],
            "val_max_top10_accuracy": best_acc["top10"],
            "val_epoch": epoch,
        }
    )

    if args.track_failure_cases:
        print(f"Average distance: {total_similarity / similarity_samples}")
        torch.save(failure_cases, f"failure_cases_{epoch}.pt")

    return best_acc
