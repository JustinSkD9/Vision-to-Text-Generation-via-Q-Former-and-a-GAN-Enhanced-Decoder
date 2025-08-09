import os
import matplotlib.pyplot as plt
import random
import json

from data.coco_utils import prepare_coco_subset
from data.dataset import load_coco_captions, create_train_val_test_splits
from PIL import Image
from collections import defaultdict

# Regenerate the dataset if changing subset size
regenerate = False

if not os.path.exists("coco_dataset_dict")  or regenerate:
    prepare_coco_subset(
        download_dir="coco_dataset",
        subset_dir="coco_subset",
        subset_size=20
    )
    dataset = load_coco_captions(subset_dir="coco_subset")
    create_train_val_test_splits(dataset, save_dir="coco_dataset_dict")
else:
    print("Dataset already prepared")


with open("coco_subset/captions.json", "r") as f:
    data = json.load(f)

# 5 captions for each image
grouped = defaultdict(list)
for item in data:
    grouped[item["image_id"]].append(item["caption"])


sample_img_id = random.choice(list(grouped.keys()))
captions = grouped[sample_img_id]

img_path = os.path.join("coco_subset/images", sample_img_id)
img = Image.open(img_path)

plt.imshow(img)
plt.axis("off")
plt.title("\n".join(captions), fontsize=9)
plt.show()