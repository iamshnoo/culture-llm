# an example script that is used to translate yahma/alpaca-cleaned dataset to hindi
import os
import random
import time

import numpy as np
import torch
from datasets import load_dataset
from process_text import is_translatable
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Set seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# args
LANG = "hindi"

MODEL_NAME = "facebook/nllb-200-1.3B"
MODEL_CACHE_DIR = "/projects/nllb-200-1.3B"

DATASET_NAME = "yahma/alpaca-cleaned"
DATASET_CACHE_DIR = "/projects/alpaca-cleaned"

langs = {
    "english": "eng_Latn",
    "hindi": "hin_Deva",
    "chinese": "zho_Hans",  # simplified chinese
    "swahili": "swh_Latn",
    "persian": "pes_Arab",  # persian
    "greek": "ell_Grek",
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=MODEL_CACHE_DIR)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, cache_dir=MODEL_CACHE_DIR)

model.to(device)

# Enable fp16 via autocasting
with torch.cuda.amp.autocast():
    dataset = load_dataset(DATASET_NAME, cache_dir=DATASET_CACHE_DIR)


def format_prompt(instruction, input=None, output=None):
    PROMPT_DICT = {
        "prompt_input": (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:{output}"
        ),
        "prompt_no_input": (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:{output}"
        ),
    }
    if input is None or input == "":
        return PROMPT_DICT["prompt_no_input"].format_map(
            {"instruction": instruction, "output": output}
        )
    else:
        return PROMPT_DICT["prompt_input"].format_map(
            {"instruction": instruction, "input": input, "output": output}
        )


def translate_text_nllb_batched(texts, source_language, target_language):
    """
    Translate the texts using the NLLB model.

    Args:
        texts (List[str]): Input texts.
        source_language (str): Source language code.
        target_language (str): Target language code.

    Returns:
        Dict[str, List[str]]: Translated texts.
    """
    # Check which texts are translatable.
    translatable_indices = [i for i, text in enumerate(texts) if is_translatable(text)]

    # Extract only the translatable texts.
    translatable_texts = [texts[i] for i in translatable_indices]

    # Tokenize all translatable texts at once.
    inputs = tokenizer(
        translatable_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=600,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate translations for all inputs at once.
    translated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id[target_language],
        max_length=800,
        early_stopping=True,
    )

    # Decode all translations at once.
    translated_texts = tokenizer.batch_decode(
        translated_tokens, skip_special_tokens=True
    )

    # Create a list to hold the translated texts.
    output_texts = texts.copy()

    # Replace the original texts with the translated texts.
    for i, translated_text in zip(translatable_indices, translated_texts):
        output_texts[i] = translated_text

    return {"text": output_texts}


start = time.time()
print("Translating...")

# # dataset has 3 columns: instruction, input, output
# translate each column separately
translated_dataset = dataset.map(
    lambda example: {
        "instruction": translate_text_nllb_batched(
            example["instruction"], langs["english"], langs[LANG]
        )["text"],
        "input": translate_text_nllb_batched(
            example["input"], langs["english"], langs[LANG]
        )["text"],
        "output": translate_text_nllb_batched(
            example["output"], langs["english"], langs[LANG]
        )["text"],
    },
    batched=True,
    batch_size=64,
)

end = time.time()
print(f"Time taken: {end-start} seconds")

print(translated_dataset["train"][0])

output_dir = f"/projects/alpaca-cleaned/translated/{LANG}"
os.makedirs(output_dir, exist_ok=True)

# save the translated dataset
translated_dataset["train"].to_csv(f"{output_dir}/train.csv")

print(f"Saved translated dataset to {output_dir}/train.csv")

# these files are uploaded to the datasets at https://huggingface.co/collections/iamshnoo/alpaca-2-64fe0c729a62bb2791f86745
