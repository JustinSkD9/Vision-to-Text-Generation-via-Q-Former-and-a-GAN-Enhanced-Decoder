import torch
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import evaluate
from tqdm import tqdm
from transformers.modeling_outputs import BaseModelOutput

meteor_metric = evaluate.load("meteor")

def evaluate_generation(qformer, decoder, dataloader, tokenizer, device="cuda"):
    qformer.eval()
    decoder.eval()

    preds, refs = [], []
    smooth_fn = SmoothingFunction().method1

    for batch in tqdm(dataloader, desc="Evaluating Metrics"):
        features = batch["features"].to(device)
        input_ids = batch["input_ids"].to(device)

        with torch.no_grad():
            vision_embeds = qformer(features)
            encoder_outputs = BaseModelOutput(last_hidden_state=vision_embeds)

            generated_ids = decoder.generate(
                encoder_outputs=encoder_outputs,
                max_length=64,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False
            )

        preds.extend([tokenizer.decode(g, skip_special_tokens=True) for g in generated_ids])
        refs.extend([tokenizer.decode(r, skip_special_tokens=True) for r in input_ids])

    bleu_score_val = corpus_bleu(
        [[r.split()] for r in refs],
        [p.split() for p in preds],
        smoothing_function=smooth_fn
    )
    rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_scores = [rouge.score(ref, pred)['rougeL'].fmeasure for ref, pred in zip(refs, preds)]
    rouge_avg = sum(rouge_scores) / len(rouge_scores)
    meteor_val = meteor_metric.compute(predictions=preds, references=refs)["meteor"]
    P, R, F1 = bert_score(preds, refs, lang="en", model_type="bert-base-uncased")
    bert_f1 = F1.mean().item()

    return bleu_score_val, rouge_avg, meteor_val, bert_f1
