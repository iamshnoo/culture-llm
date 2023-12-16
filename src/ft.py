import random
from functools import partial

import numpy as np
import torch
import wandb
from datasets import load_dataset
from llama_patch import forward, replace_attn_with_flash_attn
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
    TrainingArguments,
)
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer
import argparse

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--lang",
    type=str,
    default="english",
    choices=["english", "hindi", "chinese", "persian", "swahili", "greek"],
)
parser.add_argument(
    "--model_name", type=str, default="yi", choices=["llama-2", "yi", "uliza"]
)
parser.add_argument(
    "--llama_size", type=str, default="6", choices=["6", "7", "13", "34", "70"]
)
args = parser.parse_args()

LANG = args.lang
MODEL_NAME = args.model_name
LLAMA_SIZE = args.llama_size

print(f"{LANG}_{MODEL_NAME}_{LLAMA_SIZE}")

if LLAMA_SIZE in ["34", "70"]:
    NUM_EPOCHS = 1
else:
    NUM_EPOCHS = 3

if MODEL_NAME == "llama-2":
    wandb.init(project=f"alpaca-2-{LLAMA_SIZE}b", name=f"{LANG}-alpaca-2")
else:
    wandb.init(
        project=f"{MODEL_NAME}-alpaca-2-{LLAMA_SIZE}b",
        name=f"{LANG}-{MODEL_NAME}-alpaca-2",
    )

use_flash_attention = True if torch.cuda.get_device_capability()[0] >= 8 else False

if LLAMA_SIZE == "70":
    use_flash_attention = False

if MODEL_NAME == "llama-2":
    model_id = f"meta-llama/Llama-2-{LLAMA_SIZE}b-hf"
elif MODEL_NAME == "yi":
    model_id = f"01-ai/Yi-{LLAMA_SIZE}B"
elif MODEL_NAME == "uliza":
    model_id = "Jacaranda/kiswallama-pretrained"
MODEL_CACHE_DIR = f"/projects/{MODEL_NAME}/{LLAMA_SIZE}b"
OUTPUTS_DIR = "/projects/finetune"
MODEL_CKPT = "/projects/finetune/model"
DATASETS_CACHE_DIR = "/projects/alpaca-cleaned"

if MODEL_NAME == "yi":
    use_flash_attention = False

if use_flash_attention:
    print("Using flash attention")
    replace_attn_with_flash_attn()
    use_flash_attention = True

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    use_cache=False,
    device_map="auto",
    cache_dir=MODEL_CACHE_DIR,
)

model.config.pretraining_tp = 1

if use_flash_attention:
    assert (
        model.model.layers[0].self_attn.forward.__doc__ == forward.__doc__
    ), "Model is not using flash attention"

tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=MODEL_CACHE_DIR)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)

args = TrainingArguments(
    output_dir=OUTPUTS_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=6 if use_flash_attention else 4,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    optim="paged_adamw_32bit",
    logging_steps=10,
    save_strategy="epoch",
    learning_rate=3e-4,
    bf16=True,
    tf32=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="constant",
    disable_tqdm=True,  # disable tqdm since with packing values are in correct
)


def format_prompt(instruction, input=None, output=None, lang="english"):
    if lang == "english":
        PROMPT_DICT = {
            "prompt_input": (
                "Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}"
            ),
            "prompt_no_input": (
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n### Response:\n{output}"
            ),
        }
    elif lang == "hindi":
        PROMPT_DICT = {
            "prompt_input": (
                "नीचे एक निर्देश है जो एक कार्य का वर्णन करता है, जो एक इनपुट के साथ जोड़ा गया है जो आगे संदर्भ प्रदान करता है। "
                "एक उत्तर लिखिए जो अनुरोध को उचित रूप से पूरा करे।\n\n"
                "### निर्देश:\n{instruction}\n\n### इनपुट:\n{input}\n\n### उत्तर:\n{output}"
            ),
            "prompt_no_input": (
                "नीचे एक कार्य का वर्णन करने वाला निर्देश है। "
                "एक उत्तर लिखिए जो अनुरोध को उचित रूप से पूरा करे।\n\n"
                "### निर्देश:\n{instruction}\n\n### उत्तर:\n{output}"
            ),
        }
    elif lang == "chinese":
        PROMPT_DICT = {
            "prompt_input": (
                "下面是一个描述任务的指令,并配合一个提供进一步背景的输入. "
                "写一个适当的答案.\n\n"
                "### 指示:\n{instruction}\n\n### 输入:\n{input}\n\n### 回应:\n{output}"
            ),
            "prompt_no_input": (
                "下面是一个说明任务的指令. "
                "写一个适当的答案.\n\n"
                "### 指示:\n{instruction}\n\n### 回应:\n{output}"
            ),
        }
    elif lang == "persian":
        PROMPT_DICT = {
            "prompt_input": (
                "در زیر یک دستورالعمل است که یک کار را توصیف می کند، همراه با ورودی که زمینه بیشتری را فراهم می کند. "
                "پاسخي بنويسيد که درخواست رو به طور مناسب تکمیل کنه\n\n"
                "### آموزش:\n{instruction}\n\n### ورودی:\n{input}\n\n### پاسخ:\n{output}"
            ),
            "prompt_no_input": (
                "در زیر یک دستورالعمل است که یک کار را توصیف می کند. "
                "پاسخي بنويسيد که درخواست رو به طور مناسب تکمیل کنه\n\n"
                "### آموزش:\n{instruction}\n\n### پاسخ:\n{output}"
            ),
        }
    elif lang == "swahili":
        PROMPT_DICT = {
            "prompt_input": (
                "Chini ni amri ambayo inaelezea kazi, pamoja na input ambayo hutoa muktadha zaidi. "
                "Andika jibu linalotimiza ombi hilo.\n\n"
                "### Maagizo:\n{instruction}\n\n### Input:\n{input}\n\n### Jibu:\n{output}"
            ),
            "prompt_no_input": (
                "Chini ni maelekezo ambayo yanaelezea kazi. "
                "Andika jibu linalotimiza ombi hilo.\n\n"
                "### Maagizo:\n{instruction}\n\n### Jibu:\n{output}"
            ),
        }
    elif lang == "greek":
        PROMPT_DICT = {
            "prompt_input": (
                "Παρακάτω υπάρχει μια εντολή που περιγράφει μια εργασία, σε συνδυασμό με μια εισαγωγή που παρέχει περαιτέρω πλαίσιο. "
                "Γράψτε μια απάντηση που συμπληρώνει κατάλληλα το αίτημα.\n\n"
                "### Διδασκαλία:\n{instruction}\n\n### Εισόδου:\n{input}\n\n### Απάντηση:\n{output}"
            ),
            "prompt_no_input": (
                "Παρακάτω είναι μια οδηγία που περιγράφει μια εργασία. "
                "Γράψτε μια απάντηση που συμπληρώνει κατάλληλα το αίτημα.\n\n"
                "### Διδασκαλία:\n{instruction}\n\n### Απάντηση:\n{output}"
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


max_seq_length = 2048
if LANG == "english":
    data_path = "yahma/alpaca-cleaned"
else:
    data_path = f"iamshnoo/alpaca-cleaned-{LANG}"
dataset = load_dataset(data_path, cache_dir=DATASETS_CACHE_DIR)
dataset = dataset["train"]

dataset = dataset.map(
    lambda example: {
        "text": format_prompt(
            example["instruction"], example["input"], example["output"], lang=LANG
        ),
    },
    remove_columns=["instruction", "input", "output"],
)

splits = dataset.train_test_split(test_size=0.2)
train_dataset = splits["train"]
eval_dataset = splits["test"]

if LANG == "english":
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template=tokenizer("\n\n### Response:\n")["input_ids"][2:],
        tokenizer=tokenizer,
    )
elif LANG == "hindi":
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template=tokenizer("\n\n### उत्तर:\n")["input_ids"][2:],
        tokenizer=tokenizer,
    )
elif LANG == "chinese":
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template=tokenizer("\n\n### 回应:\n")["input_ids"][2:],
        tokenizer=tokenizer,
    )
elif LANG == "persian":
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template=tokenizer("### پاسخ:\n")["input_ids"][2:],
        tokenizer=tokenizer,
    )
elif LANG == "swahili":
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template=tokenizer("\n\n### Jibu:\n")["input_ids"][2:],
        tokenizer=tokenizer,
    )
elif LANG == "greek":
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template=tokenizer("\n\n### Απάντηση:\n")["input_ids"][2:],
        tokenizer=tokenizer,
    )

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    dataset_text_field="text",
    data_collator=data_collator,
    peft_config=peft_config,
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=args,
)

trainer.train()
trainer.save_model(MODEL_CKPT)

if MODEL_NAME == "llama-2":
    model.push_to_hub(f"iamshnoo/alpaca-2-{LLAMA_SIZE}b-{LANG}")
elif MODEL_NAME == "yi":
    model.push_to_hub(f"iamshnoo/yi-alpaca-2-{LLAMA_SIZE}b-{LANG}")
elif MODEL_NAME == "uliza":
    model.push_to_hub(f"iamshnoo/uliza-alpaca-2-{LLAMA_SIZE}b-{LANG}")
