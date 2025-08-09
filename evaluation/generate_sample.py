import random
import torch
import matplotlib.pyplot as plt
from transformers.modeling_outputs import BaseModelOutput
from PIL import Image

def show_generated_examples(qformer, decoder, tokenizer, dataset_dict, features_tensor, device="cuda", num_examples=5, start_idx=0):
    qformer.eval()
    decoder.eval()

    # select indices for examples
    sample_indices = list(range(start_idx, min(start_idx + num_examples, len(dataset_dict["validation"]))))

    for idx in sample_indices:
        # Load image and reference caption
        img_path = dataset_dict["validation"][idx]["image_path"]
        caption_ref = dataset_dict["validation"][idx]["caption"]

        img = Image.open(img_path).convert("RGB")

        # Extract features for the image
        feature = features_tensor[idx].unsqueeze(0).to(device)  # [1, clip_dim]

        with torch.no_grad():
            vision_embeds = qformer(feature)  # [1, num_query_tokens, hidden_size]
            encoder_outputs = BaseModelOutput(last_hidden_state=vision_embeds)

            generated_ids = decoder.generate(
                encoder_outputs=encoder_outputs,
                max_length=64,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False,
                repetition_penalty=2.0,
                no_repeat_ngram_size=3
            )

        caption_pred = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        # Display the image and captions
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"Pred: {caption_pred}\nRef:  {caption_ref}", fontsize=9)
        plt.show()

