import torch

def load_checkpoint(path, qformer, decoder, optimizer=None, device="cuda"):
    checkpoint = torch.load(path, map_location=device)
    qformer.load_state_dict(checkpoint['qformer_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    best_bert = checkpoint.get('bert_score', 0)
    
    print(f"Loaded checkpoint from epoch {epoch} with BERTScore {best_bert}")
    return epoch, best_bert
