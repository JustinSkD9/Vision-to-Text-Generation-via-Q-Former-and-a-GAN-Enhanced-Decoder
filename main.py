import torch
from datasets import load_from_disk
from features.extract_clip import extract_clip_patch_features
from training.train_supervised import train_supervised
from utils.load_checkpoint import load_checkpoint, load_gan_checkpoint
from transformers import T5Tokenizer
from models.qformer import QFormerEncoder
from models.t5_decoder import T5ForConditionalGeneration
from models.test_gan import TestGAN
from training.train_gan import train_gan
from config.path import BASE_DIR
from evaluation.generate_sample import show_generated_examples
import os
import json
import argparse


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Vision-to-Text Generation Training')
    parser.add_argument('--use-gan', action='store_true', default=False,
                        help='Enable GAN training after supervised training')
    parser.add_argument('--skip-supervised', action='store_true', default=False,
                        help='Skip supervised training (use existing checkpoint)')
    parser.add_argument('--re-extract', action='store_true', default=False,
                        help='Re-extract CLIP features')
    parser.add_argument('--checkpoint', action='store_true', default=False,
                        help='Load existing supervised checkpoint if available')
    parser.add_argument('--gan-checkpoint', action='store_true', default=False,
                        help='Load existing GAN checkpoint if available')
    parser.add_argument('--supervised-epochs', type=int, default=30,
                        help='Number of supervised training epochs')
    parser.add_argument('--gan-epochs', type=int, default=20,
                        help='Number of GAN training epochs')
    parser.add_argument('--warmup-epochs', type=int, default=5,
                        help='Number of warmup epochs (T5 frozen)')
    parser.add_argument('--examples', type=int, default=5,
                        help='Number of examples to generate')
    
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load dataset dict
    print("Loading dataset...")
    dataset_dict = load_from_disk("coco_dataset_dict")

    # Extract clip features once if not done yet/ or load if already extracted
    if not os.path.exists("clip_features/train_clip.pt") or args.re_extract:
        print("Extracting CLIP features...")
        extract_clip_patch_features(dataset_dict, save_dir="clip_features", device=device)

    # Load features
    print("Loading features...")
    train_features = torch.load("clip_features/train_clip.pt")
    val_features = torch.load("clip_features/validation_clip.pt")

    # Initialize models and optimizer
    print("Initializing models...")
    qformer = QFormerEncoder(input_dim=train_features.shape[2], hidden_size=768).to(device)
    decoder = T5ForConditionalGeneration.from_pretrained("t5-base").to(device)
    optimizer = torch.optim.Adam(list(qformer.parameters()) + list(decoder.parameters()), lr=1e-4)

    checkpoint_path = "best_model_bert.pt"
    gan_checkpoint_path = "best_model_gan.pt"

    if os.path.exists(checkpoint_path) and args.checkpoint:
        print(f"Loading checkpoint from {checkpoint_path}")
        # If checkpoint exists, load model and optimizer states before training
        load_checkpoint(
            checkpoint_path, qformer, decoder, device=device
        )

    # Supervised training
    if not args.skip_supervised:
        print(f"Starting supervised training for {args.supervised_epochs} epochs...")
        # Now run training starting from start_epoch
        history, qformer, decoder, optimizer = train_supervised(
            dataset_dict,
            train_features,
            val_features,
            device=device,
            save_path="best_model_bert.pt",
            total_epochs=args.supervised_epochs,
            warmup_epochs=args.warmup_epochs,
        )

        # Export training history to a local file
        with open("training_history.json", "w") as f:
            json.dump(history, f)
        print("Supervised training history saved to training_history.json")
    else:
        print("Skipping supervised training...")

    # GAN training
    if args.use_gan:
        print(f"Starting GAN training for {args.gan_epochs} epochs...")
        
        # Check if GAN checkpoint exists and we want to resume GAN training
        if os.path.exists(gan_checkpoint_path) and args.gan_checkpoint:
            print(f"Found existing GAN checkpoint: {gan_checkpoint_path}")
            
            # Get tokenizer first to determine vocab size
            tokenizer = T5Tokenizer.from_pretrained("t5-base")
            
            # Load checkpoint to get the actual vocab size used during training
            checkpoint_info = torch.load(gan_checkpoint_path, map_location=device)
            
            # Try to get vocab size from checkpoint, fallback to tokenizer if not available
            if 'discriminator_state_dict' in checkpoint_info:
                embed_weight_shape = checkpoint_info['discriminator_state_dict']['embed.weight'].shape
                checkpoint_vocab_size = embed_weight_shape[0]
                print(f"Checkpoint discriminator vocab size: {checkpoint_vocab_size}")
            else:
                checkpoint_vocab_size = len(tokenizer)
                print(f"Using tokenizer vocab size: {checkpoint_vocab_size}")
            
            # Initialize discriminator with the correct vocab size
            discriminator = TestGAN(vocab_size=checkpoint_vocab_size).to(device)
            optimizer_G = torch.optim.Adam(list(qformer.parameters()) + list(decoder.parameters()), lr=1e-4)
            optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=1e-4)
            
            # Load GAN checkpoint using dedicated function
            epoch, best_bert = load_gan_checkpoint(
                gan_checkpoint_path, qformer, decoder, discriminator, 
                optimizer_G, optimizer_D, device=device
            )
            
            # Continue GAN training for additional epochs if specified
            if args.gan_epochs > 0:
                print(f"Continuing GAN training for {args.gan_epochs} more epochs...")
                history_gan, qformer, decoder, discriminator, optimizer_G, optimizer_D, tokenizer = train_gan(
                    dataset_dict,
                    train_features,
                    val_features,
                    device=device,
                    num_epochs=args.gan_epochs,
                    save_path="best_model_gan.pt"
                )

                # Export training history to a local file
                with open("training_history_gan.json", "w") as f:
                    json.dump(history_gan, f)
                print("GAN training history saved to training_history_gan.json")
            else:
                print("No additional GAN training requested (gan-epochs = 0)")
            
        else:
            print("No GAN checkpoint found or checkpoint loading disabled. Starting GAN training from scratch...")
            history_gan, qformer, decoder, discriminator, optimizer_G, optimizer_D, tokenizer = train_gan(
                dataset_dict,
                train_features,
                val_features,
                device=device,
                num_epochs=args.gan_epochs,
                save_path="best_model_gan.pt"
            )

            # Export training history to a local file
            with open("training_history_gan.json", "w") as f:
                json.dump(history_gan, f)
            print("GAN training history saved to training_history_gan.json")
    else:
        print("Skipping GAN training...")
        # Create tokenizer for example generation if GAN was skipped
        tokenizer = T5Tokenizer.from_pretrained("t5-base")

    # Generate examples
    print(f"Generating {args.examples} example captions...")
    show_generated_examples(qformer, decoder, tokenizer, dataset_dict, val_features, device=device, num_examples=args.examples)
    print("Training complete!")


if __name__ == "__main__":
    main()