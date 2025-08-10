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

def load_gan_checkpoint(path, qformer, decoder, discriminator, optimizer_G=None, optimizer_D=None, device="cuda"):
    checkpoint = torch.load(path, map_location=device)
    
    # Load model states
    qformer.load_state_dict(checkpoint['qformer_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    
    # Load optimizer states if provided
    if optimizer_G and 'optimizer_G_state_dict' in checkpoint:
        optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
    
    if optimizer_D and 'optimizer_D_state_dict' in checkpoint:
        optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
    
    # Get training metadata
    epoch = checkpoint.get('epoch', 0)
    best_bert = checkpoint.get('bert_score', 0)
    
    print(f"Loaded GAN checkpoint from epoch {epoch} with BERTScore {best_bert:.4f}")
    print(f"Restored: QFormer, T5 Decoder, Discriminator, and Optimizers")
    
    return epoch, best_bert
