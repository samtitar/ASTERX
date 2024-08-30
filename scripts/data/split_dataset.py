import os
import argparse
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str)
    parser.add_argument("--level", type=str, choices=["panel", "character"], default="panel")
    args = parser.parse_args()

    comics = os.listdir(f"{args.data_dir}/panel_images")

    train = comics[: int(0.8 * len(comics))]
    val = comics[int(0.8 * len(comics)) : int(0.9 * len(comics))]
    test = comics[int(0.9 * len(comics)) :]

    splits = {"train": train, "val": val, "test": test}

    for split in ["train", "val", "test"]:
        if not os.path.exists(f"{args.data_dir}/{split}_images"):
            os.mkdir(f"{args.data_dir}/{split}_images")
            
        for comic in splits[split]:
            if os.path.exists(f"{args.data_dir}/{args.level}_images/{comic}"):
                os.system(
                    "cp -r "
                    + f"{args.data_dir}/{args.level}_images/{comic} "
                    + f"{args.data_dir}/{split}_images/"
                )
