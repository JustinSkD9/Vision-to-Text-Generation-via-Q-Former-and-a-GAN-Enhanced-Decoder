import torch

BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 5e-5
MAX_LEN = 32
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print
