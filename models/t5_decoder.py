from transformers import T5ForConditionalGeneration
import torch.optim as optim

def get_t5_decoder(model_name="t5-base"):
    return T5ForConditionalGeneration.from_pretrained(model_name)

def get_t5_decoder_and_optimizer(qformer, lr=1e-4, device="cuda"):
    decoder = T5ForConditionalGeneration.from_pretrained("t5-base").to(device)
    # Optimize both qformer and decoder parameters
    optimizer = optim.Adam(list(qformer.parameters()) + list(decoder.parameters()), lr=lr)
    return decoder, optimizer