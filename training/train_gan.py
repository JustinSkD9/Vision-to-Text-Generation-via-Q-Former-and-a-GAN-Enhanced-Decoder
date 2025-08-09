import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import T5Tokenizer
from transformers.modeling_outputs import BaseModelOutput
from models.test_gan import TestGAN  # renamed discriminator class file
from training.eval import evaluate_generation

def train_gan(dataset_dict, train_features, val_features, device="cuda", num_epochs=10, save_path=None):
    import torch.nn as nn
    import torch.optim as optim

    tokenizer = T5Tokenizer.from_pretrained("t5-base")

    # Create tokenized datasets (same as train_supervised)
    train_tokens = tokenizer([x["caption"] for x in dataset_dict["train"]], padding="max_length", truncation=True, max_length=32, return_tensors="pt")
    val_tokens = tokenizer([x["caption"] for x in dataset_dict["validation"]], padding="max_length", truncation=True, max_length=32, return_tensors="pt")

    class QFormerTextDataset(torch.utils.data.Dataset):
        def __init__(self, features, tokens):
            self.features = features
            self.input_ids = tokens.input_ids
            self.attention_mask = tokens.attention_mask

        def __len__(self):
            return len(self.features)

        def __getitem__(self, idx):
            return {
                "features": self.features[idx],
                "input_ids": self.input_ids[idx],
                "attention_mask": self.attention_mask[idx]
            }

    train_dataset = QFormerTextDataset(train_features, train_tokens)
    val_dataset = QFormerTextDataset(val_features, val_tokens)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)

    # Initialize models (same as train_supervised)
    from models.qformer import QFormerEncoder
    from models.t5_decoder import get_t5_decoder
    
    qformer = QFormerEncoder(input_dim=train_features.shape[2], hidden_size=768).to(device)
    decoder = get_t5_decoder().to(device)
    import torch.nn as nn
    import torch.optim as optim

    bce_loss = nn.BCEWithLogitsLoss()
    ce_loss = nn.CrossEntropyLoss(ignore_index=-100)

    discriminator = TestGAN(vocab_size=tokenizer.vocab_size).to(device)

    optimizer_G = optim.Adam(list(qformer.parameters()) + list(decoder.parameters()), lr=1e-4)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=1e-4)

    best_bert = 0.0
    history = {"epoch": [], "bleu": [], "rouge": [], "bert_f1": [], "meteor": [], 
               "g_loss": [], "d_loss": [], "ce_loss": [], "gan_loss": [], "val_loss": []}

    for epoch in range(num_epochs):
        qformer.train()
        decoder.train()
        discriminator.train()

        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        epoch_ce_loss = 0.0
        epoch_gan_loss = 0.0
        num_batches = 0

        for batch in tqdm(train_loader, desc=f"Train Epoch {epoch}"):
            features = batch["features"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = input_ids.clone()
            labels[labels == tokenizer.pad_token_id] = -100  # fix incomplete line

            # Q-Former encoding
            encoder_hidden = qformer(features)
            encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden)

            # CE loss for decoder
            outputs = decoder(
                encoder_outputs=encoder_outputs,
                labels=labels,
                attention_mask=attention_mask
            )
            ce_loss_val = outputs.loss

            # Generate without teacher forcing for GAN discriminator
            with torch.no_grad():
                gen_ids = decoder.generate(
                    encoder_outputs=encoder_outputs,
                    max_length=64,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )

            # Discriminator labels
            real_labels = torch.ones(input_ids.size(0), 1).to(device)
            fake_labels = torch.zeros(input_ids.size(0), 1).to(device)

            # Discriminator forward pass
            real_logits = discriminator(input_ids, attention_mask)
            fake_logits = discriminator(gen_ids, gen_ids != tokenizer.pad_token_id)

            # Discriminator losses
            d_loss_real = bce_loss(real_logits, real_labels)
            d_loss_fake = bce_loss(fake_logits, fake_labels)
            d_loss = (d_loss_real + d_loss_fake) / 2

            optimizer_D.zero_grad()
            d_loss.backward()
            optimizer_D.step()

            # Generator GAN loss
            fake_logits_g = discriminator(gen_ids, gen_ids != tokenizer.pad_token_id)
            gan_loss_val = bce_loss(fake_logits_g, real_labels)

            # Total generator loss
            g_total_loss = ce_loss_val + 0.1 * gan_loss_val

            optimizer_G.zero_grad()
            g_total_loss.backward()
            optimizer_G.step()

            # Track losses for this batch
            epoch_g_loss += g_total_loss.item()
            epoch_d_loss += d_loss.item()
            epoch_ce_loss += ce_loss_val.item()
            epoch_gan_loss += gan_loss_val.item()
            num_batches += 1

        # Calculate average losses for the epoch
        avg_g_loss = epoch_g_loss / num_batches
        avg_d_loss = epoch_d_loss / num_batches
        avg_ce_loss = epoch_ce_loss / num_batches
        avg_gan_loss = epoch_gan_loss / num_batches
        
        print(f"Epoch {epoch} | G Loss: {avg_g_loss:.4f} | D Loss: {avg_d_loss:.4f} | "
              f"CE Loss: {avg_ce_loss:.4f} | GAN Loss: {avg_gan_loss:.4f}")

        # Validation
        qformer.eval()
        decoder.eval()
        discriminator.eval()
        val_loss = 0.0
        total_batches = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                features = batch["features"].to(device)
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = input_ids.clone()
                labels[labels == tokenizer.pad_token_id] = -100

                encoder_hidden = qformer(features)
                encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden)
                outputs = decoder(
                    encoder_outputs=encoder_outputs,
                    labels=labels,
                    attention_mask=attention_mask
                )
                val_loss += outputs.loss.item()
                total_batches += 1

        avg_val_loss = val_loss / total_batches
        print(f"Epoch {epoch} | Validation Loss: {avg_val_loss:.4f}")

        # Validation metrics
        bleu, rouge, meteor, bert_f1 = evaluate_generation(qformer, decoder, val_loader, tokenizer, device)
        print(f"Epoch {epoch} | BLEU: {bleu:.4f} | ROUGE-L: {rouge:.4f} | "
              f"BERTScore-F1: {bert_f1:.4f} | Meteor: {meteor:.4f}")

        # Save best model based on BERTScore F1
        if bert_f1 > best_bert and save_path:
            best_bert = bert_f1
            torch.save({
                'epoch': epoch,
                'qformer_state_dict': qformer.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
                'bert_score': best_bert
            }, save_path)
            print(f"New best model saved at epoch {epoch} with BERTScore {best_bert:.4f}")

        # Update history
        history["epoch"].append(epoch)
        history["bleu"].append(bleu)
        history["rouge"].append(rouge)
        history["bert_f1"].append(bert_f1)
        history["meteor"].append(meteor)
        history["g_loss"].append(avg_g_loss)
        history["d_loss"].append(avg_d_loss)
        history["ce_loss"].append(avg_ce_loss)
        history["gan_loss"].append(avg_gan_loss)
        history["val_loss"].append(avg_val_loss)

    return history, qformer, decoder, discriminator, optimizer_G, optimizer_D, tokenizer
