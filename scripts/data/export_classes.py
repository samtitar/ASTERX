import os
import json
import argparse
import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str)
    parser.add_argument("--target-column", type=str, default="Global Region")
    args = parser.parse_args()

    df = pd.read_csv(f"{args.data_dir}/annotations/metadata.csv")

    comic_to_cls = {}
    cls_to_label = {}

    for comic in os.listdir(f"{args.data_dir}/panel_images"):
        if comic not in df["Title Norm"].unique():
            continue

        series = df[df["Title Norm"] == comic][args.target_column].astype(str).str.lower()

        if len(series) == 0:
            continue
        elif len(series) > 1:
            series = series.iloc[:1]

        label = series.item()
        if label not in cls_to_label:
            cls_to_label[label] = len(cls_to_label)
        comic_to_cls[comic] = cls_to_label[label]

    with open(f"{args.data_dir}/classification.json", "w") as f:
        json.dump({"comic_to_cls": comic_to_cls, "cls_to_label": cls_to_label}, f)
