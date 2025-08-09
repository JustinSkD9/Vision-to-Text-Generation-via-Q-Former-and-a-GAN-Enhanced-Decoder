import os
from PIL import Image
import torch
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

def extract_clip_patch_features(dataset_dict, save_dir, batch_size=8, device="cuda"):
    os.makedirs(save_dir, exist_ok=True)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device).eval()

    for split in dataset_dict:
        save_path = os.path.join(save_dir, f"{split}_clip.pt")
        print(f"Extracting patch features for {split}...")

        image_paths = [x["image_path"] for x in dataset_dict[split]]
        split_features = []

        for i in tqdm(range(0, len(image_paths), batch_size)):
            batch_paths = image_paths[i:i+batch_size]
            batch_images = [Image.open(p).convert("RGB") for p in batch_paths]

            inputs = clip_processor(images=batch_images, return_tensors="pt", padding=True).to(device)

            with torch.no_grad():
                vision_outputs = clip_model.vision_model(inputs["pixel_values"])
                patch_tokens = vision_outputs.last_hidden_state[:, 1:, :]  # remove CLS token
            split_features.append(patch_tokens.cpu())

        features_tensor = torch.cat(split_features, dim=0)
        print(features_tensor.shape)
        torch.save(features_tensor, save_path)
        print(f"Saved {split} patch features to {save_path}")
