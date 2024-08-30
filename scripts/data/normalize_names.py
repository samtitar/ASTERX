import os
import re
import argparse
import pandas as pd

normalize_name = (
    lambda x: re.sub(
        r"['.,!?;:()\[\]{}\/éç…]",
        "-",
        x.lower(),
    )
    .replace(" ", "_")
    .replace('"', "-")
    .replace("&", "-")
    .replace("#", "-")
    .replace("--", "-")
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str)
    args = parser.parse_args()

    # Normalize file names
    names = os.listdir(f"{args.data_dir}/page_images")
    for name in names:
        norm_name = normalize_name(name)
        os.rename(
            f"{args.data_dir}/page_images/{name}",
            f"{args.data_dir}/page_images/{norm_name}",
        )

    # Normalize panel annotations
    df = pd.read_csv(f"{args.data_dir}/annotations/panels.csv")
    df["Document Name"] = df["Document Name"].apply(normalize_name)
    df.to_csv(f"{args.data_dir}/annotations/panels.csv", index=False)

    # # Normalize emotion annotations
    df = pd.read_csv(f"{args.data_dir}/annotations/emotions.csv")
    df["Document Name"] = df["Document Name"].apply(normalize_name)
    df.to_csv(f"{args.data_dir}/annotations/emotions.csv", index=False)

    # Normalize character annotations
    df = pd.read_csv(f"{args.data_dir}/annotations/characters.csv")
    df["Document Name"] = df["Document Name"].apply(normalize_name)
    df.to_csv(f"{args.data_dir}/annotations/characters.csv", index=False)

    # Normalize metadata annotations
    df_ref = pd.read_csv(f"{args.data_dir}/annotations/panels.csv")

    # Use df_ref to create mapping from "Docuument Directory" to "Document Name"
    doc_dir_to_name = {}
    for _, row in df_ref.iterrows():
        doc_dir_to_name[row["Document Directory"]] = row["Document Name"]

    # Apply mapping to metadata
    df = pd.read_csv(f"{args.data_dir}/annotations/metadata.csv")
    df["Title Norm"] = df["Document Directory"].apply(lambda x: doc_dir_to_name[x])
    df.to_csv(f"{args.data_dir}/annotations/metadata.csv", index=False)
