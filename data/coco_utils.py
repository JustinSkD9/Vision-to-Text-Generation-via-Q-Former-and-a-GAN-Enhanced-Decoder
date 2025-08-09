import os
import zipfile
import random
import json
import shutil
from tqdm import tqdm
from pycocotools.coco import COCO
import urllib.request

# for windows compatibility
def download_file(url, dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)  # Ensure folder exists
    if not os.path.exists(dest):
        print(f"Downloading {url} ...")
        urllib.request.urlretrieve(url, dest)
    else:
        print(f"{dest} already exists, skipping download.")

def prepare_coco_subset(download_dir="coco_dataset", subset_dir="coco_subset", subset_size=10000):
    os.makedirs(download_dir, exist_ok=True)

    train_zip = os.path.join(download_dir, "train2017.zip")
    ann_zip = os.path.join(download_dir, "annotations_trainval2017.zip")

    # Download on Linux/Mac
    # os.system("wget -nc http://images.cocodataset.org/zips/train2017.zip")
    # os.system("wget -nc http://images.cocodataset.org/annotations/annotations_trainval2017.zip")

    # Download on Windows
    download_file("http://images.cocodataset.org/zips/train2017.zip", train_zip)
    download_file("http://images.cocodataset.org/annotations/annotations_trainval2017.zip", ann_zip)
    
    os.chdir(download_dir)

    # Extract
    with zipfile.ZipFile("train2017.zip", "r") as zip_ref:
        zip_ref.extractall("images")
    with zipfile.ZipFile("annotations_trainval2017.zip", "r") as zip_ref:
        zip_ref.extractall("annotations")

    os.chdir("..")

    ann_path = os.path.join(download_dir, "annotations/annotations/captions_train2017.json")
    coco = COCO(ann_path)
    all_image_ids = coco.getImgIds()
    print("COCO total images:", len(all_image_ids))

    subset_image_ids = random.sample(all_image_ids, subset_size)

    image_dir = os.path.join(subset_dir, "images")
    os.makedirs(image_dir, exist_ok=True)

    captions = []

    for img_id in tqdm(subset_image_ids, desc="Copying images and captions"):
        img_info = coco.loadImgs(img_id)[0]
        file_name = img_info['file_name']
        src_path = os.path.join(download_dir, f"images/train2017/{file_name}")
        dst_path = os.path.join(image_dir, file_name)
        shutil.copyfile(src_path, dst_path)

        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        for ann in anns:
            captions.append({"image_id": file_name, "caption": ann["caption"]})

    with open(os.path.join(subset_dir, "captions.json"), "w") as f:
        json.dump(captions, f)

    print(f"Captions count: {len(captions)}")
    print(f"Images count: {len(subset_image_ids)}")
