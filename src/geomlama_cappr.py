import random
import json
import gc

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from peft import PeftConfig, PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from no_country_prompts import get_no_country_questions
from cappr.huggingface.classify import predict_proba

# random seed settings
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

LANG_TO_CODE = {
    "english": "en",
    "hindi": "hi",
    "chinese": "zh",
    "swahili": "sw",
    "persian": "fa",
    "greek": "el",
}

DATASETS_CACHE_DIR = "/projects/geomlama"


def format_prompt(instruction, input="", output="", lang="english"):
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


with open("geomlama_configs.json") as f:
    all_combinations = json.load(f)

for exp_id, args in tqdm(
    all_combinations.items(), total=len(all_combinations), desc="Experiment"
):
    MODEL_NAME = args[0]
    LLAMA_SIZE = args[1]
    LANG = args[2]
    MODE = args[3]
    print(f"{LANG}_{LLAMA_SIZE}_{MODE}_{MODEL_NAME}")

    MODEL_CACHE_DIR = f"/projects/{MODEL_NAME}/{LLAMA_SIZE}b"
    dataset = load_dataset("iamshnoo/geomlama", cache_dir=DATASETS_CACHE_DIR)
    dataset = dataset[LANG_TO_CODE[LANG]]
    no_country_df = get_no_country_questions(LANG_TO_CODE[LANG])

    if MODEL_NAME == "uliza-alt":
        model_id = "Jacaranda/UlizaLlama"
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            return_dict=True,
            load_in_4bit=True,
            device_map="auto",
            cache_dir=MODEL_CACHE_DIR,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=MODEL_CACHE_DIR,
        )

    else:
        peft_model_id = (
            f"iamshnoo/{MODEL_NAME}-alpaca-2-{LLAMA_SIZE}b-{LANG}"
            if MODEL_NAME != "alpaca-2"
            else f"iamshnoo/alpaca-2-{LLAMA_SIZE}b-{LANG}"
        )

        config = PeftConfig.from_pretrained(peft_model_id)
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path,
            return_dict=True,
            load_in_4bit=True,
            device_map="auto",
            cache_dir=MODEL_CACHE_DIR,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            config.base_model_name_or_path,
            cache_dir=MODEL_CACHE_DIR,
        )
        model = PeftModel.from_pretrained(
            model,
            peft_model_id,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            load_in_4bit=True,
            device_map="auto",
            cache_dir=MODEL_CACHE_DIR,
        )

    df = pd.DataFrame(
        columns=["question", "options", "correct_answer", "response", "probabilities"]
    )

    for i in tqdm(range(len(dataset["question"]))):
        if MODE == "country":
            question = dataset["question"][i]
        else:
            if i % 6 == 0:
                question = no_country_df["question"][i]
            else:
                continue

        context = dataset["context"][i]
        options = dataset["candidate_answers"][i]
        correct_answer = dataset["answer"][i]

        split_options = options.split(",")
        split_options = [o.strip() for o in split_options]
        random.shuffle(split_options)
        options_formatted = "\n".join(
            [f"({chr(i+65)}) {o}" for i, o in enumerate(split_options)]
        )

        if LANG == "english":
            instruction = "You are an expert in global cultures and the subject matter of the question. Using that expertise, identify the most accurate answer from the options provided for the given question."
            input = f"Question: {question} \nOptions: {options_formatted}"
        elif LANG == "hindi":
            instruction = "आपको हिंदी में ही जवाब देना है। आप वैश्विक संस्कृतियों और प्रश्न के विषय में एक विशेषज्ञ हैं। अपनी विशेषज्ञता का इस्तेमाल करके, दिए गए प्रश्न के लिए दिए गए विकल्पों में से सबसे सटीक उत्तर का चयन करें और हिंदी में ही प्रतिक्रिया दें।"
            input = f"प्रश्न: {question} \nविकल्प: {options_formatted}"
        elif LANG == "chinese":
            instruction = "您是全球文化和问题主题的专家。 请使用您的专业知识从所提供的选项中识别出最准确的答案，并用中文回答。"
            input = f"问题: {question} \n选项: {options_formatted}"
        elif LANG == "swahili":
            instruction = "Wewe ni mtaalam wa tamaduni za ulimwengu na somo la swali. Kwa kutumia utaalamu huo, pata jibu sahihi zaidi kutoka kwa chaguo ulizopewa kwa swali lililotolewa. Jibu kwa swahili."
            input = f"Swali: {question} \nChaguo: {options_formatted}"
        elif LANG == "persian":
            instruction = "شما یک متخصص فرهنگ های جهانی و موضوع سوال هستید. با استفاده از آن تخصص، پاسخ دقیق تر را از گزینه های ارائه شده برای سوال مشخص شده شناسایی کنید و به فارسی پاسخ دهید."
            input = f"سوال: {question} \nگزینه ها: {options_formatted}"
        elif LANG == "greek":
            instruction = "Είστε ειδικός στον παγκόσμιο πολιτισμό και το θέμα της ερώτησης. Χρησιμοποιώντας αυτήν την ειδικότητα, εντοπίστε την πιο ακριβή απάντηση από τις παρεχόμενες επιλογές για την δοθείσα ερώτηση και απαντήστε στα ελληνικά."
            input = f"Ερώτηση: {question} \nΕπιλογές: {options_formatted}"

        prompt = format_prompt(instruction, input=input, lang=LANG)
        pred_probs = predict_proba(
            prompt,
            completions=split_options,
            model_and_tokenizer=(model, tokenizer),
            end_of_prompt="",
            batch_size=1,
        )
        pred_probs_rounded = pred_probs.round(2)
        pred_probs_rounded = dict(zip(split_options, pred_probs_rounded))
        pred_probs_rounded = {
            k: v
            for k, v in sorted(
                pred_probs_rounded.items(), key=lambda item: item[1], reverse=True
            )
        }
        answer = split_options[np.argmax(pred_probs)]

        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    [[question, options, correct_answer, answer, pred_probs_rounded]],
                    columns=[
                        "question",
                        "options",
                        "correct_answer",
                        "response",
                        "probabilities",
                    ],
                ),
            ],
            ignore_index=True,
        )

        print("Question: ", question)
        print("Options:")
        print(options_formatted)
        print("Correct Answer: ", correct_answer)
        print("Probabilities: ", pred_probs_rounded)
        print("Model Answer: ", answer)
        print("-" * 80)

    PATH = f"../outputs/geomlama/{MODE}/{LANG}_results_{MODEL_NAME}_{LLAMA_SIZE}b.csv"
    print(f"Experiment {exp_id} completed. Saving results to {PATH}")
    if MODE == "no_country":
        # correct answer would be country specific (fig 5 of geomlama paper)
        df = df.drop(columns=["correct_answer"])
    df.to_csv(PATH, index=False)
    gc.collect()
