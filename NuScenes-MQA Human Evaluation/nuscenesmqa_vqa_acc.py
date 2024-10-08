import sys
sys.path.insert(0, '/home/nikostheodoridis/PhD/Paper/Main')

import pandas as pd

from nuscenes import NuScenes
from PIL import Image
from tqdm import tqdm
from IPython.display import clear_output, display


def human_evaluation(num_examples: int = 100,
                     dataset_path: str = "/home/nikostheodoridis/nuscenes-mqa/test_set.csv",
                     nuscenes_path: str = "/home/nikostheodoridis/nuscenes"):
    test_set = pd.read_csv(dataset_path, dtype="str").iloc[:, 1:]

    nusc = NuScenes(version="v1.0-trainval", dataroot=nuscenes_path, verbose=False)

    score = 0

    for _ in tqdm(range(num_examples)):
        print(f"Score: {score}")
        sample_row = test_set.sample()

        test_set.drop(sample_row.index)

        sample = nusc.get("sample", sample_row["sample_token"].iloc[0])

        for sensor, token in sample["data"].items():
            if sensor[:3] == "CAM":
                print(sensor)
                sample_data = nusc.get("sample_data", token)
                filename = f"{nuscenes_path}/{sample_data['filename']}"
                display(Image.open(filename))
                # nusc.render_sample_data(token, with_anns=False)

        answer = input(sample_row["question"].iloc[0])

        if answer == sample_row["answer"].iloc[0]:
            score += 1

        clear_output()

    print(f"Your Accuracy: {(score / num_examples) * 100}%")
