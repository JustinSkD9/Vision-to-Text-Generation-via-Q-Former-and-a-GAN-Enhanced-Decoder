import torch
from datasets import load_from_disk
from features.extract_clip import extract_clip_patch_features
from training.train_supervised import train_supervised
from utils.load_checkpoint import load_checkpoint
from transformers import T5Tokenizer
from models.qformer import QFormerEncoder
from models.t5_decoder import T5ForConditionalGeneration
from models.test_gan import TestGAN
from training.train_gan import train_gan
from config.path import BASE_DIR
from evaluation.generate_sample import show_generated_examples
import os
import json


device = "cuda" if torch.cuda.is_available() else "cpu"

# Load dataset dict
dataset_dict = load_from_disk("coco_dataset_dict")

re_extract = False  # Set to True to re-extract features

# Extract clip features once if not done yet/ or load if already extracted
if not os.path.exists("clip_features/train_clip.pt") or re_extract:
    extract_clip_patch_features(dataset_dict, save_dir="clip_features", device=device)

# Load features
train_features = torch.load("clip_features/train_clip.pt")
val_features = torch.load("clip_features/validation_clip.pt")

# Initialize models and optimizer
qformer = QFormerEncoder(input_dim=train_features.shape[2], hidden_size=768).to(device)
decoder = T5ForConditionalGeneration.from_pretrained("t5-base").to(device)
optimizer = torch.optim.Adam(list(qformer.parameters()) + list(decoder.parameters()), lr=1e-4)

checkpoint_path = "best_model_bert.pt"


checkpoint = True  # Set to True if you want to load a checkpoint

if os.path.exists(checkpoint_path) and checkpoint:
    # If checkpoint exists, load model and optimizer states before training
    load_checkpoint(
        checkpoint_path, qformer, decoder, device="cuda"
    )

# Now run training starting from start_epoch
history, qformer, decoder, optimizer = train_supervised(
    dataset_dict,
    train_features,
    val_features,
    device=device,
    save_path="best_model_bert.pt",
    total_epochs=5,
    warmup_epochs=3,
)

# Export training history to a local file
with open("training_history.json", "w") as f:
    json.dump(history, f)

use_gan = True  # or True if you want to train with GAN

if use_gan:
    history_gan, qformer, decoder, discriminator, optimizer_G, optimizer_D, tokenizer = train_gan(
        dataset_dict,
        train_features,
        val_features,
        device=device,
        num_epochs=10,
        save_path="best_model_gan.pt"
    )

# Export training history to a local file
with open("training_history_gan.json", "w") as f:
    json.dump(history_gan, f)

# Create tokenizer for example generation
show_generated_examples(qformer, decoder, tokenizer, dataset_dict, val_features, device="cuda", num_examples=5)