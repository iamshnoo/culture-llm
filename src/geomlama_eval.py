import pandas as pd
import numpy as np
import json
from tqdm import tqdm

COUNTRIES = ["US", "China", "India", "Iran", "Kenya", "Greece"]
LANGS = ["english", "chinese", "hindi", "persian", "swahili", "greek"]
LLAMA_SIZES = ["6", "7", "13", "34", "70"]


def eval_country(filename):
    scores = {country: 0 for country in COUNTRIES}

    df = pd.read_csv(filename)  # 150 rows
    filename_no_country = filename.replace("country", "no_country")
    df_no_country = pd.read_csv(filename_no_country)  # 25 rows

    # Split the correct answers by comma and remove whitespace
    df["correct_answer"] = df["correct_answer"].apply(lambda x: x.split(","))
    df["correct_answer"] = df["correct_answer"].apply(
        lambda x: [ans.strip() for ans in x]
    )
    # add a country column to the dataframe
    df["country"] = COUNTRIES * 25

    for country in COUNTRIES:
        country_df = df[df["country"] == country]
        country_c = 0  # corrects
        country_g = 0  # golds
        for idx, row in country_df.iterrows():
            idx_no_country = idx // 6
            row_no_country = df_no_country.iloc[idx_no_country]

            correct_answer = row["correct_answer"]
            probabilities = eval(row["probabilities"])
            probabilities_no_country = eval(row_no_country["probabilities"])
            for key in probabilities.keys():
                probabilities[key] -= probabilities_no_country[key]

            # Case 1: if len(correct_answer) == 1, then extract probabilities[0]
            if len(correct_answer) == 1:
                if correct_answer[0] == list(probabilities.keys())[0]:
                    country_c += 1
                country_g += 1
            # Case 2: if len(correct_answer) > 1, then extract probabilities[:len(correct_answer)]
            else:
                # extract the top len(correct_answer) probabilities
                top_probs = list(probabilities.keys())[: len(correct_answer)]
                # check if the top probabilities are in the correct answers
                for ans in top_probs:
                    if ans in correct_answer:
                        country_c += 1
                country_g += len(correct_answer)

        # calculate the score for the country
        country_score = country_c / country_g
        scores[country] = country_score
    return scores


def eval_no_country(filename):
    scores = {country: 0 for country in COUNTRIES}

    df = pd.read_csv(filename)  # 25 rows
    # create an empty column for the correct answer
    # df["correct_answer"] = np.nan
    filename_country = filename.replace("no_country", "country")
    df_country = pd.read_csv(filename_country)  # 150 rows
    df_country["country"] = COUNTRIES * 25

    # the correct answer would depend on the country chosen
    for country in COUNTRIES:
        df["country"] = country
        df_country_one_country = df_country[df_country["country"] == country]
        correct_answers = df_country_one_country["correct_answer"].tolist()
        df["correct_answer"] = correct_answers

        # Split the correct answers by comma and remove whitespace
        df["correct_answer"] = df["correct_answer"].apply(lambda x: x.split(","))
        df["correct_answer"] = df["correct_answer"].apply(
            lambda x: [ans.strip() for ans in x]
        )

        country_c = 0  # corrects
        country_g = 0  # golds

        for idx, row in df.iterrows():
            correct_answer = row["correct_answer"]
            probabilities = eval(row["probabilities"])
            # Case 1: if len(correct_answer) == 1, then extract probabilities[0]
            if len(correct_answer) == 1:
                if correct_answer[0] == list(probabilities.keys())[0]:
                    country_c += 1
                country_g += 1
            # Case 2: if len(correct_answer) > 1, then extract probabilities[:len(correct_answer)]
            else:
                # extract the top len(correct_answer) probabilities
                top_probs = list(probabilities.keys())[: len(correct_answer)]
                # check if the top probabilities are in the correct answers
                for ans in top_probs:
                    if ans in correct_answer:
                        country_c += 1
                country_g += len(correct_answer)

        # calculate the score for the country
        country_score = country_c / country_g
        scores[country] = country_score

    return scores


def filter_df(df, filters):
    # apply consecutive filters to the dataframe
    for key, value in filters.items():
        df = df[df[key] == value]
    return df


def extract(key, component, options):
    # key is a str
    # component is a str
    # options is a list of str for the component
    if component == "mode":
        if "country" in key and "no_country" not in key:
            return "country"
        elif "no_country" in key:
            return "no_country"
    if component == "model_name":
        if "alpaca-2" in key:
            return "alpaca-2"
        elif "yi" in key:
            return "yi"
        elif "uliza" in key and "uliza-alt" not in key:
            return "uliza"
        elif "uliza-alt" in key:
            return "uliza-alt"
    if component == "llama_size":
        if "6" in key:
            return "6"
        elif "7" in key and "70" not in key:
            return "7"
        elif "13" in key:
            return "13"
        elif "34" in key:
            return "34"
        elif "70" in key:
            return "70"
    for option in options:
        if option in key:
            return option


if __name__ == "__main__":
    with open("geomlama_configs.json") as f:
        all_combinations = json.load(f)

    results = {}
    for exp_id, args in tqdm(
        all_combinations.items(), total=len(all_combinations), desc="Experiment"
    ):
        MODEL_NAME = args[0]
        LLAMA_SIZE = args[1]
        LANG = args[2]
        MODE = args[3]
        key = f"{LANG}_{LLAMA_SIZE}_{MODE}_{MODEL_NAME}"
        filename = (
            f"../outputs/geomlama/{MODE}/{LANG}_results_{MODEL_NAME}_{LLAMA_SIZE}b.csv"
        )
        if MODE == "country":
            scores = eval_country(filename)
            results[key] = scores
        elif MODE == "no_country":
            scores = eval_no_country(filename)
            results[key] = scores

    df = pd.DataFrame(results)
    df = df.transpose()
    df = df.rename_axis("key").reset_index()

    for i, row in df.iterrows():
        df.at[i, "lang"] = extract(row["key"], "lang", LANGS)
        df.at[i, "llama_size"] = extract(row["key"], "llama_size", LLAMA_SIZES)
        df.at[i, "mode"] = extract(row["key"], "mode", ["country", "no_country"])
        df.at[i, "model_name"] = extract(
            row["key"], "model_name", ["alpaca-2", "yi", "uliza", "uliza-alt"]
        )

    df_alpaca_2_country = pd.concat(
        [
            group
            for _, group in filter_df(
                df, {"mode": "country", "model_name": "alpaca-2"}
            ).groupby("lang")
        ]
    )
    df_alpaca_2_no_country = pd.concat(
        [
            group
            for _, group in filter_df(
                df, {"mode": "no_country", "model_name": "alpaca-2"}
            ).groupby("lang")
        ]
    )
    df_yi_country = pd.concat(
        [
            group
            for _, group in filter_df(
                df, {"mode": "country", "model_name": "yi"}
            ).groupby("lang")
        ]
    )
    df_yi_no_country = pd.concat(
        [
            group
            for _, group in filter_df(
                df, {"mode": "no_country", "model_name": "yi"}
            ).groupby("lang")
        ]
    )
    df_uliza_country = pd.concat(
        [
            group
            for _, group in filter_df(
                df, {"mode": "country", "model_name": "uliza"}
            ).groupby("lang")
        ]
    )
    df_uliza_no_country = pd.concat(
        [
            group
            for _, group in filter_df(
                df, {"mode": "no_country", "model_name": "uliza"}
            ).groupby("lang")
        ]
    )
    df_uliza_alt_country = pd.concat(
        [
            group
            for _, group in filter_df(
                df, {"mode": "country", "model_name": "uliza-alt"}
            ).groupby("lang")
        ]
    )
    df_uliza_alt_no_country = pd.concat(
        [
            group
            for _, group in filter_df(
                df, {"mode": "no_country", "model_name": "uliza-alt"}
            ).groupby("lang")
        ]
    )
    df_swahili_country = pd.concat(
        [
            group
            for _, group in filter_df(
                df, {"mode": "country", "lang": "swahili"}
            ).groupby("model_name")
        ]
    )
    df_chinese_country = pd.concat(
        [
            group
            for _, group in filter_df(
                df, {"mode": "country", "lang": "chinese"}
            ).groupby("model_name")
        ]
    )

    export_path = f"../outputs/geomlama/evals"
    df.to_csv(f"{export_path}/all_results.csv", index=False)

    df_alpaca_2_country.to_csv(f"{export_path}/alpaca_2_country.csv", index=False)
    df_alpaca_2_no_country.to_csv(f"{export_path}/alpaca_2_no_country.csv", index=False)
    df_yi_country.to_csv(f"{export_path}/yi_country.csv", index=False)
    df_yi_no_country.to_csv(f"{export_path}/yi_no_country.csv", index=False)
    df_uliza_country.to_csv(f"{export_path}/uliza_country.csv", index=False)
    df_uliza_no_country.to_csv(f"{export_path}/uliza_no_country.csv", index=False)
    df_uliza_alt_country.to_csv(f"{export_path}/uliza_alt_country.csv", index=False)
    df_uliza_alt_no_country.to_csv(
        f"{export_path}/uliza_alt_no_country.csv", index=False
    )
    df_swahili_country.to_csv(f"{export_path}/swahili_country.csv", index=False)
    df_chinese_country.to_csv(f"{export_path}/chinese_country.csv", index=False)

    # Main findings:
    print(df_alpaca_2_country)
    print(df_swahili_country[["key", "US", "Kenya"]])
    print(df_chinese_country[["key", "US", "China"]])
