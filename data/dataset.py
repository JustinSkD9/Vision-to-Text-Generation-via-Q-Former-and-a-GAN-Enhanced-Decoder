import os
import json
from datasets import Dataset, DatasetDict
import random

def load_coco_captions(subset_dir="coco_subset"):
    with open(os.path.join(subset_dir, "captions.json"), "r") as f:
        raw_data = json.load(f)

    data = {"image_path": [], "caption": []}
    for item in raw_data:
        img_path = os.path.join(subset_dir, "images", item["image_id"])
        if os.path.exists(img_path):
            data["image_path"].append(img_path)
            data["caption"].append(item["caption"])

    dataset = Dataset.from_dict(data)
    print(f"Dataset size: {len(dataset)}")
    return dataset

def create_train_val_test_splits(dataset, save_dir, seed=42):
    train_test = dataset.train_test_split(test_size=0.2, seed=seed)
    val_test = train_test["test"].train_test_split(test_size=0.5, seed=seed)
    dataset_dict = DatasetDict({
        "train": train_test["train"],
        "validation": val_test["train"],
        "test": val_test["test"]
    })
    dataset_dict.save_to_disk(save_dir)
    print(f"DatasetDict saved to {save_dir}")
    for split in dataset_dict:
        print(f"{split}: {len(dataset_dict[split])}")
    return dataset_dict

def create_smaller_subset(dataset_dict, fraction=0.01, save_dir=None, seed=42):
    from datasets import DatasetDict
    small_dataset_dict = DatasetDict()
    for split in dataset_dict:
        small_dataset_dict[split] = dataset_dict[split].train_test_split(
            test_size=1 - fraction, seed=seed)['train']
    if save_dir:
        small_dataset_dict.save_to_disk(save_dir)
        print(f"Small DatasetDict saved to {save_dir}")
    for split in small_dataset_dict:
        print(f"{split}: {len(small_dataset_dict[split])}")
    return small_dataset_dict
