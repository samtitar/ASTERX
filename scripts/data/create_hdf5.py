import os
import io
import h5py
import json
import argparse
import numpy as np

from PIL import Image
from tqdm import tqdm


UNDERSCORE_SORT = lambda s: list(map(try_int, s.split("_")))


def try_int(s):
    try:
        return int(s)
    except ValueError:
        return int.from_bytes(s.encode(), "little")


def process_dir(path, name, writer):
    contents = os.listdir(path)
    grp = writer.create_group(name)

    path_index = {}
    img_list, img_idx = [], []
    str_list, str_idx = [], []

    for content in sorted(contents, key=UNDERSCORE_SORT):
        content_l = content.lower()
        if os.path.isdir(f"{path}/{content}"):
            path_index[content] = process_dir(f"{path}/{content}", content, grp)
        elif (
            content_l.endswith(".jpg")
            or content_l.endswith(".jpeg")
            or content_l.endswith(".png")
        ):
            entry_name = content.split(".")[0]
            with open(f"{path}/{content}", "rb") as f:
                img_data = f.read()

            img_list.append(img_data)
            img_idx.append(entry_name)
        elif content_l.endswith(".csv"):
            entry_name = content.split(".")[0]
            with open(f"{path}/{content}", "r") as f:
                csv_data = f.read()

            str_list.append(csv_data)
            str_idx.append(entry_name)

    if len(img_list) > 0:
        path_index["img_data"] = img_idx
        dt = h5py.special_dtype(vlen=np.dtype("uint8"))
        grp.create_dataset(
            "img_data",
            (len(img_list),),
            dtype=dt,
        )
        for i, img in enumerate(img_list):
            grp["img_data"][i] = np.frombuffer(img, dtype="uint8")

    if len(str_list) > 0:
        path_index["str_data"] = str_idx
        dt = h5py.special_dtype(vlen=str)
        grp.create_dataset(
            "str_data",
            (len(str_list),),
            dtype=dt,
        )
        for i, csv in enumerate(str_list):
            grp["str_data"][i] = csv

    return path_index


def main(args):
    path = args.data_path
    dataset_name = path.split("/")[-1]
    writer = h5py.File(f"{path}.hdf5", "w")

    index = {}
    index[dataset_name] = process_dir(path, dataset_name, writer)

    writer.close()

    with open(f"{path}_indexing.json", "w+") as f:
        json.dump(index, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert directories to hdf5")
    parser.add_argument("--data-path", type=str)
    args = parser.parse_args()
    main(args)
