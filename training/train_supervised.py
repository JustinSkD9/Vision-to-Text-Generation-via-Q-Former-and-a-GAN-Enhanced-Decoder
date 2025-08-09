import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import T5Tokenizer
from models.qformer import QFormerEncoder
from models.t5_decoder import get_t5_decoder
from training.eval import evaluate_generation

def freeze_all_t5(decoder):
    for param in decoder.parameters():
        param.requires_grad = False

def unfreeze_all_t5(decoder):
    for param in decoder.parameters():
        param.requires_grad = True

def train_supervised(dataset_dict, train_features, val_features, device="cuda", save_path=None, total_epochs=20, warmup_epochs=3):
    tokenizer = T5Tokenizer.from_pretrained("t5-base")

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

    qformer = QFormerEncoder(input_dim=train_features.shape[2], hidden_size=768).to(device)
    decoder = get_t5_decoder().to(device)

    optimizer = torch.optim.Adam(list(qformer.parameters()) + list(decoder.parameters()), lr=1e-4)

    freeze_all_t5(decoder)

    best_bert = 0.0
    history = {"epoch": [], "bleu": [], "rouge": [], "bert_f1": [], "val_loss": [], "meteor": []}

    for epoch in range(total_epochs):
        if epoch == warmup_epochs:
            print(f"===> Stage 2: Unfreezing T5 cross-attention & LayerNorm at epoch {epoch}")
            unfreeze_all_t5(decoder)

        decoder.train()
        qformer.train()
        for batch in tqdm(train_loader, desc=f"Train Epoch {epoch}"):
            features = batch["features"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = input_ids.clone()
            labels[labels == tokenizer.pad_token_id] = -100

            vision_embeds = qformer(features)
            outputs = decoder(
                encoder_outputs=(vision_embeds,),
                labels=labels,
                attention_mask=attention_mask
            )

            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

        # Validation
        decoder.eval()
        qformer.eval()
        val_loss = 0.0
        total_batches = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                features = batch["features"].to(device)
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = input_ids.clone()
                labels[labels == tokenizer.pad_token_id] = -100

                vision_embeds = qformer(features)
                outputs = decoder(
                    encoder_outputs=(vision_embeds,),
                    labels=labels,
                    attention_mask=attention_mask
                )
                val_loss += outputs.loss.item()
                total_batches += 1

        avg_val_loss = val_loss / total_batches
        print(f"Epoch {epoch} | Validation Loss: {avg_val_loss:.4f}")

        bleu, rouge, meteor, bert_f1 = evaluate_generation(qformer, decoder, val_loader, tokenizer, device)
        print(f"Epoch {epoch} | BLEU: {bleu:.4f} | ROUGE-L: {rouge:.4f} | "
              f"BERTScore-F1: {bert_f1:.4f} | Meteor: {meteor:.4f}")

        if bert_f1 > best_bert and save_path:
            best_bert = bert_f1
            torch.save({
                'epoch': epoch,
                'qformer_state_dict': qformer.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'bert_score': best_bert
            }, save_path)
            print(f"New best model saved at epoch {epoch} with BERTScore {best_bert:.4f}")

        history["epoch"].append(epoch)
        history["bleu"].append(bleu)
        history["rouge"].append(rouge)
        history["bert_f1"].append(bert_f1)
        history["val_loss"].append(avg_val_loss)
        history["meteor"].append(meteor)

    return history, qformer, decoder, optimizer
