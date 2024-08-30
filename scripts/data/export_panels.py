import os
import argparse
import numpy as np
import pandas as pd

from PIL import Image
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str)
    args = parser.parse_args()

    df = pd.read_csv(f"{args.data_dir}/annotations/panels.csv")

    if not os.path.exists(f"{args.data_dir}/panel_images"):
        os.mkdir(f"{args.data_dir}/panel_images")

    comics = df["Document Name"].unique()

    pbar = tqdm(total=len(df))

    for comic in comics:
        comic_df = df[df["Document Name"] == comic]
        pages = sorted(comic_df["Page Number"].unique())

        if not os.path.exists(f"{args.data_dir}/panel_images/{comic}"):
            os.mkdir(f"{args.data_dir}/panel_images/{comic}")

        for page in pages:
            page_df = comic_df[comic_df["Page Number"] == page]
            page_df = page_df.sort_values(by=["Annotation Notes"])

            if not os.path.exists(
                f"{args.data_dir}/page_images/{comic}/page-{page}.jpg"
            ):
                continue

            for i, row in page_df.iterrows():
                panel_coords = row["Region Vertices"]
                panel_coords = panel_coords.split(",")
                coords = [
                    int(x.replace("(", "").replace(")", "")) for x in panel_coords
                ]

                if len(coords) > 4:
                    # Convert polygon to bounding box by taking min/max of x/y coords
                    coords = np.array(coords).reshape(-1, 2)
                    coords = np.concatenate([coords.min(axis=0), coords.max(axis=0)])
                    coords = coords.tolist()

                panel = Image.open(
                    f"{args.data_dir}/page_images/{comic}/page-{page}.jpg"
                ).crop(coords)

                panel_num = row["Annotation Notes"]
                panel.save(
                    f"{args.data_dir}/panel_images/{comic}/{page}_{panel_num}.jpg"
                )

                pbar.update(1)
