import json
import math
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from scipy.special import softmax

MODEL_NAME = "distilgpt2"
SAMPLES_PER_PROMPT = 5
MAX_NEW_TOKENS = 40

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.eval()

def entropy_from_logits(logits):
    probs = softmax(logits, axis=-1)
    return -sum(p * math.log(p + 1e-12) for p in probs)

def generate_answer(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            do_sample=True,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.8,
            output_scores=True,
            return_dict_in_generate=True
        )

    generated_tokens = outputs.sequences[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # entropy from first generated token (proxy confidence)
    first_step_logits = outputs.scores[0][0].numpy()
    ent = entropy_from_logits(first_step_logits)

    return text.strip(), ent

def run_experiment(prompt_file, condition):
    results = []

    with open(prompt_file, "r") as f:
        prompts = json.load(f)

    for item in prompts:
        question = item["question"]
        correct = item["answer"]
        context = item["context"]

        full_prompt = f"{context}\n\nQuestion: {question}\nAnswer:"

        for _ in range(SAMPLES_PER_PROMPT):
            answer, ent = generate_answer(full_prompt)

            results.append({
                "condition": condition,
                "question": question,
                "model_answer": answer,
                "correct": any(word in answer.lower() for word in correct.lower().split()),
                "entropy": ent
            })

    return results

all_results = []
all_results += run_experiment("prompts/clean.json", "clean")
all_results += run_experiment("prompts/corrupted.json", "corrupted")
all_results += run_experiment("prompts/missing.json", "missing")

df = pd.DataFrame(all_results)
df.to_csv("results.csv", index=False)

print("Experiment complete. Results saved to results.csv")
