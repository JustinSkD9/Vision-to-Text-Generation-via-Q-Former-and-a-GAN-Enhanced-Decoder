import json
import os
from evaluation.generate_sample import show_generated_examples
from evaluation.plot_history import plot_history
from models.t5_decoder import T5ForConditionalGeneration
from models.qformer import QFormerEncoder
    

# Load history.json
history_path = os.path.join(os.path.dirname(__file__), 'training_history.json')
with open(history_path, 'r') as f:
    history = json.load(f)

# Plot the training history
plot_history(history, 'Without GAN')

history_path = os.path.join(os.path.dirname(__file__), 'training_history_gan.json')
with open(history_path, 'r') as f:
    history = json.load(f)

# Plot the training history
plot_history(history, 'With GAN')
