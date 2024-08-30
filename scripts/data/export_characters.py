import os
import json
import argparse
import numpy as np
import pandas as pd

from PIL import Image
from tqdm import tqdm


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

    characters_df = pd.read_csv(f"{args.data_dir}/annotations/characters.csv")
    emotions_df = pd.read_csv(f"{args.data_dir}/annotations/emotions.csv")
    panels_df = pd.read_csv(f"{args.data_dir}/annotations/panels.csv")
    result = []

    if not os.path.exists(f"{args.data_dir}/character_images"):
        os.mkdir(f"{args.data_dir}/character_images")

    for i, row in tqdm(characters_df.iterrows(), total=len(characters_df)):
        document_name = row["Document Name"]

        if type(row["Region Vertices"]) != str:
            continue

        page = row["Region Page"]

        character_coords = row["Region Vertices"]
        character_coords = character_coords.split(",")
        character_coords = [
            int(x.replace("(", "").replace(")", "")) for x in character_coords
        ]

        if len(character_coords) > 4:
            # Convert polygon to bounding box by taking min/max of x/y coords
            character_coords = np.array(character_coords).reshape(-1, 2)
            character_coords = np.concatenate(
                [character_coords.min(axis=0), character_coords.max(axis=0)]
            )
            character_coords = character_coords.tolist()

        candidate_panels = panels_df[
            (panels_df["Document Name"] == document_name)
            & (panels_df["Page Number"] == page)
        ]

        for j, candidate_panel in candidate_panels.iterrows():
            panel_coords = candidate_panel["Region Vertices"]
            panel_coords = panel_coords.split(",")
            panel_coords = [
                int(x.replace("(", "").replace(")", "")) for x in panel_coords
            ]

            if len(panel_coords) > 4:
                # Convert polygon to bounding box by taking min/max of x/y coords
                panel_coords = np.array(panel_coords).reshape(-1, 2)
                panel_coords = np.concatenate(
                    [panel_coords.min(axis=0), panel_coords.max(axis=0)]
                )
                panel_coords = panel_coords.tolist()

            if (
                character_coords[0] >= panel_coords[0]
                and character_coords[1] >= panel_coords[1]
                and character_coords[2] <= panel_coords[2]
                and character_coords[3] <= panel_coords[3]
            ):
                character = row["Relation ID"]
                panel = candidate_panel["Annotation Notes"]

                panel_id = f"{int(page)}_{panel}"

                # Convert rectangle to panel coordinates
                character_coords = np.array(character_coords).reshape(-1, 2)
                character_coords[:, 0] -= panel_coords[0]
                character_coords[:, 1] -= panel_coords[1]
                character_coords = character_coords.reshape(-1).tolist()

                # Extract arousal and valence
                region_id = row["Region ID"]
                emotion_rows = emotions_df[emotions_df["Region ID"] == region_id]
                valence = emotion_rows[
                    emotion_rows["Taxonomy Path"].str.contains("Valence")
                ]["Taxonomy Path"].values
                arousal = emotion_rows[
                    emotion_rows["Taxonomy Path"].str.contains("Arousal")
                ]["Annotation Notes"].values

                if len(valence) == 0:
                    valence =  -1
                else:
                    valence = valence[0]

                if len(arousal) == 0:
                    arousal = -1
                else:
                    arousal = arousal[0]

                valence_map = {
                    "VLT: Semantics: Emotion (v.4) / Valence / Negative": 1,
                    "VLT: Semantics: Emotion (v.4) / Valence / Slighly Negative": 2,
                    "VLT: Semantics: Emotion (v.4) / Valence / Neutral": 3,
                    "VLT: Semantics: Emotion (v.4) / Valence / Slightly Positive": 4,
                    "VLT: Semantics: Emotion (v.4) / Valence / Positive": 5,
                }

                valence = valence_map.get(valence, -1)
                try:
                    arousal = int(arousal)
                except ValueError:
                    arousal = -1

                result.append(
                    {
                        "character": f"{document_name}_{character}",
                        "document": document_name,
                        "page": page,
                        "panel": panel_id,
                        "panel_id": f"{document_name}/{panel_id}",
                        "x0": character_coords[0],
                        "y0": character_coords[1],
                        "x1": character_coords[2],
                        "y1": character_coords[3],
                        "valence": valence,
                        "arousal": arousal,
                    }
                )

                if not os.path.exists(
                    f"{args.data_dir}/character_images/{document_name}"
                ):
                    os.mkdir(f"{args.data_dir}/character_images/{document_name}")

                character_image = Image.open(
                    f"{args.data_dir}/panel_images/{document_name}/{panel_id}.jpg"
                ).crop(character_coords)

                character_image.save(
                    f"{args.data_dir}/character_images/{document_name}/{panel_id}_{character}.jpg"
                )

    result = pd.DataFrame(result)

    character_frequency = result["character"].value_counts()
    result["character_frequency"] = result["character"].map(
        lambda x: character_frequency[x]
    )

    result.to_csv(f"{args.data_dir}/annotations/characters_indexing.csv", index=False)
