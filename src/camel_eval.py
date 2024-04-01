import pandas as pd
import numpy as np
from tqdm import tqdm
import json

prompts_path = "../data/camel/prompts.json"
targets_path = "../data/camel/targets.json"

with open(prompts_path) as f:
    PROMPTS = json.load(f)

with open(targets_path) as f:
    TARGETS = json.load(f)


# **NOTE**, for camel dataset, we are using "USA", whereas for geomlama we are doing "US"
COUNTRIES = ["USA", "China", "India", "Iran", "Kenya", "Greece"]
CATEGORIES = [
    "female_names",
    "male_names",
    "food",
    "female_clothing",
    "male_clothing",
    "location",
    "literature",
    "beverage",
    "religion",
]
GENDER_CATEGORIES = ["female_names", "male_names", "female_clothing", "male_clothing"]
LLAMA_SIZES = ["7", "13", "70"]


## TYPE 1 ##
def type1_results_helper(path):
    df = pd.read_csv(path)
    results = pd.DataFrame(columns=COUNTRIES, index=CATEGORIES)
    for country in COUNTRIES:
        for category in CATEGORIES:
            df_filtered = df[df["country"] == country]
            df_filtered = df_filtered[df_filtered["category"] == category]
            accuracy = 0
            for idx, row in df_filtered.iterrows():
                if row["correct"] == row["response"]:
                    accuracy += 1
            accuracy /= len(df_filtered)
            results[country][category] = accuracy
    return results


def type1_results_helper_non_granular(path):
    df = pd.read_csv(path)
    # the result will be a single row of accuracies for each country
    results = pd.DataFrame(columns=COUNTRIES, index=["Overall"])
    for country in COUNTRIES:
        df_filtered = df[df["country"] == country]
        accuracy = 0
        for idx, row in df_filtered.iterrows():
            if row["correct"] == row["response"]:
                accuracy += 1
        accuracy /= len(df_filtered)
        results[country] = accuracy
    return results


def type1_results(path):
    all_results = []

    for LLAMA_SIZE in LLAMA_SIZES:
        path1 = path.format(LLAMA_SIZE=LLAMA_SIZE)
        results1 = type1_results_helper(path1)
        overall_results1 = type1_results_helper_non_granular(path1)
        results1 = pd.concat([results1, overall_results1])
        results1["Llama_Size"] = LLAMA_SIZE
        all_results.append(results1)

    result = pd.concat(all_results)
    # rename index to category
    result = result.rename_axis("Category").reset_index()
    result["Llama_Size"] = result["Llama_Size"].astype(int)
    result = result.sort_values(by=["Category", "Llama_Size"])
    # move the llama size column to after the category column
    llama_size = result.pop("Llama_Size")
    result.insert(1, "Llama_Size", llama_size)
    # round all the cols to 2 decimal places
    for col in result.columns:
        if col not in ["Category", "Llama_Size"]:
            result[col] = result[col].apply(lambda x: round(x, 2))
    # reset index
    result = result.reset_index(drop=True)
    result.to_csv(
        "../outputs/camel/evals/english_results_alpaca-2_type_1.csv", index=False
    )
    return result


## TYPE 2 ##
def find_response_country(sample):
    targets_df = pd.DataFrame(TARGETS)
    targets_df = targets_df.rename_axis("category").reset_index()
    targets_df = targets_df[targets_df["category"] == sample["category"]]
    for col in targets_df.columns:
        if col == sample["country"]:
            targets_df = targets_df.drop(columns=[col])
    targets_df = targets_df.drop(columns=["category"])
    response_country = ""
    for col in targets_df.columns:
        if sample["response"] in targets_df[col].values[0]:
            response_country = col
            break
    return response_country


def type2_results_helper(df):
    results = pd.DataFrame(columns=COUNTRIES, index=CATEGORIES)
    for country in COUNTRIES:
        for category in CATEGORIES:
            df_filtered = df[df["country"] == country]
            df_filtered = df_filtered[df_filtered["category"] == category]
            counts = {}
            for c in COUNTRIES:
                counts[c] = 0
            for idx, row in df_filtered.iterrows():
                counts[row["response_country"]] += 1
            total = len(df_filtered)
            for c in COUNTRIES:
                counts[c] /= total
                counts[c] = round(counts[c], 2)
            results[country][category] = counts
    return results


def type2_results_helper_non_granular(df):
    # in this don't need to be at the granular level of category
    # results will be countries x countries, like a heatmap
    results = pd.DataFrame(columns=COUNTRIES, index=COUNTRIES)
    # aggregate counts across all categories for each country
    # then divide by total number of samples for that country
    for country in COUNTRIES:
        df_filtered = df[df["country"] == country]
        counts = {}
        for c in COUNTRIES:
            counts[c] = 0
        for idx, row in df_filtered.iterrows():
            counts[row["response_country"]] += 1
        total = len(df_filtered)
        for c in COUNTRIES:
            counts[c] /= total
            counts[c] = round(counts[c], 2)
        results[country] = counts
    return results.T


def type2_results(path):
    all_results = []

    for LLAMA_SIZE in LLAMA_SIZES:
        path2 = path.format(LLAMA_SIZE=LLAMA_SIZE)
        df = pd.read_csv(path2)
        df["response_country"] = df.apply(
            lambda row: find_response_country(row), axis=1
        )
        results = type2_results_helper(df)
        results["Llama_Size"] = LLAMA_SIZE
        all_results.append(results)

    result = pd.concat(all_results)
    result = result.rename_axis("Category").reset_index()
    result["Llama_Size"] = result["Llama_Size"].astype(int)
    result = result.sort_values(by=["Category", "Llama_Size"])
    # move the llama size column to after the category column
    llama_size = result.pop("Llama_Size")
    result.insert(1, "Llama_Size", llama_size)
    # reset index
    result = result.reset_index(drop=True)
    result.to_csv(
        "../outputs/camel/evals/english_results_alpaca-2_type_2.csv", index=False
    )
    return result


def type2_results_non_granular(path):
    all_results = []

    for LLAMA_SIZE in LLAMA_SIZES:
        path2 = path.format(LLAMA_SIZE=LLAMA_SIZE)
        df = pd.read_csv(path2)
        df["response_country"] = df.apply(
            lambda row: find_response_country(row), axis=1
        )
        results = type2_results_helper_non_granular(df)
        results["Llama_Size"] = LLAMA_SIZE
        all_results.append(results)

    result = pd.concat(all_results)
    result = result.rename_axis("Prompt").reset_index()
    result["Llama_Size"] = result["Llama_Size"].astype(int)
    result = result.sort_values(by=["Prompt", "Llama_Size"])
    # move the llama size column to after the prompt column
    llama_size = result.pop("Llama_Size")
    result.insert(1, "Llama_Size", llama_size)
    # reset index
    result = result.reset_index(drop=True)
    result.to_csv(
        "../outputs/camel/evals/non_granular/english_results_alpaca-2_type_2_non_granular.csv",
        index=False,
    )
    return result


## TYPE 3 ##
def type3_results_helper(df):
    results = pd.DataFrame(columns=COUNTRIES, index=CATEGORIES)
    for country in COUNTRIES:
        for category in CATEGORIES:
            df_filtered = df[df["country"] == country]
            df_filtered = df_filtered[df_filtered["category"] == category]
            incorrects = 0
            for idx, row in df_filtered.iterrows():
                if row["response"] in row["incorrect"]:
                    incorrects += 1
            inaccuracy = incorrects / len(df_filtered)
            results[country][category] = inaccuracy
    return results


def type3_results_helper_non_granular(df):
    results = pd.DataFrame(columns=COUNTRIES, index=["Overall"])
    for country in COUNTRIES:
        df_filtered = df[df["country"] == country]
        incorrects = 0
        for idx, row in df_filtered.iterrows():
            if row["response"] in row["incorrect"]:
                incorrects += 1
        inaccuracy = incorrects / len(df_filtered)
        results[country] = inaccuracy
    return results


def type3_results(path):
    all_results = []

    for LLAMA_SIZE in LLAMA_SIZES:
        path3 = path.format(LLAMA_SIZE=LLAMA_SIZE)
        df = pd.read_csv(path3)
        results = type3_results_helper(df)
        overall_results = type3_results_helper_non_granular(df)
        results = pd.concat([results, overall_results])
        results["Llama_Size"] = LLAMA_SIZE
        all_results.append(results)

    result = pd.concat(all_results)
    result = result.rename_axis("Category").reset_index()
    result["Llama_Size"] = result["Llama_Size"].astype(int)
    result = result.sort_values(by=["Category", "Llama_Size"])
    # move the llama size column to after the category column
    llama_size = result.pop("Llama_Size")
    result.insert(1, "Llama_Size", llama_size)
    # round all the cols to 2 decimal places
    for col in result.columns:
        if col not in ["Category", "Llama_Size"]:
            result[col] = result[col].apply(lambda x: round(x, 2))
    # reset index
    result = result.reset_index(drop=True)
    result.to_csv(
        "../outputs/camel/evals/english_results_alpaca-2_type_3.csv", index=False
    )
    return result


## TYPE 4 ##
# (same eval strategy as type 3, as we are checking how many times the response is in incorrect)
def type4_results(path):
    all_results = []

    for LLAMA_SIZE in LLAMA_SIZES:
        path3 = path.format(LLAMA_SIZE=LLAMA_SIZE)
        df = pd.read_csv(path3)
        results = type3_results_helper(df)
        overall_results = type3_results_helper_non_granular(df)
        results = pd.concat([results, overall_results])
        results["Llama_Size"] = LLAMA_SIZE
        all_results.append(results)

    result = pd.concat(all_results)
    result = result.rename_axis("Category").reset_index()
    result["Llama_Size"] = result["Llama_Size"].astype(int)
    result = result.sort_values(by=["Category", "Llama_Size"])
    # move the llama size column to after the category column
    llama_size = result.pop("Llama_Size")
    result.insert(1, "Llama_Size", llama_size)
    # round all the cols to 2 decimal places
    for col in result.columns:
        if col not in ["Category", "Llama_Size"]:
            result[col] = result[col].apply(lambda x: round(x, 2))
    # reset index
    result = result.reset_index(drop=True)
    result.to_csv(
        "../outputs/camel/evals/english_results_alpaca-2_type_4.csv", index=False
    )
    return result


## TYPE 5 ##
# (same eval strategy as type 3, 4, as we are checking how many times the response is in incorrect)


def type5_results_helper(df):
    results = pd.DataFrame(columns=COUNTRIES, index=GENDER_CATEGORIES)
    for country in COUNTRIES:
        for category in GENDER_CATEGORIES:
            df_filtered = df[df["country"] == country]
            df_filtered = df_filtered[df_filtered["category"] == category]
            incorrects = 0
            for idx, row in df_filtered.iterrows():
                if row["response"] in row["incorrect"]:
                    incorrects += 1
            inaccuracy = incorrects / len(df_filtered)
            results[country][category] = inaccuracy
    return results


def type5_results(path):
    all_results = []

    for LLAMA_SIZE in LLAMA_SIZES:
        path3 = path.format(LLAMA_SIZE=LLAMA_SIZE)
        df = pd.read_csv(path3)
        results = type5_results_helper(df)
        overall_results = type3_results_helper_non_granular(df)
        results = pd.concat([results, overall_results])
        results["Llama_Size"] = LLAMA_SIZE
        all_results.append(results)

    result = pd.concat(all_results)
    result = result.rename_axis("Category").reset_index()
    result["Llama_Size"] = result["Llama_Size"].astype(int)
    result = result.sort_values(by=["Category", "Llama_Size"])
    # move the llama size column to after the category column
    llama_size = result.pop("Llama_Size")
    result.insert(1, "Llama_Size", llama_size)
    # round all the cols to 2 decimal places
    for col in result.columns:
        if col not in ["Category", "Llama_Size"]:
            result[col] = result[col].apply(lambda x: round(x, 2))
    # reset index
    result = result.reset_index(drop=True)
    result.to_csv(
        "../outputs/camel/evals/english_results_alpaca-2_type_5.csv", index=False
    )
    return result


if __name__ == "__main__":
    # path1 = "../outputs/camel/type1/english_results_alpaca-2_{LLAMA_SIZE}b_type_1.csv"
    # print(type1_results(path1))
    # path2 = "../outputs/camel/type2/english_results_alpaca-2_{LLAMA_SIZE}b_type_2.csv"
    # print(type2_results_non_granular(path2))
    # print(type2_results(path2))
    # path3 = "../outputs/camel/type3/english_results_alpaca-2_{LLAMA_SIZE}b_type_3.csv"
    # print(type3_results(path3))
    path4 = "../outputs/camel/type4/english_results_alpaca-2_{LLAMA_SIZE}b_type_4.csv"
    print(type4_results(path4))
    # path5 = "../outputs/camel/type5/english_results_alpaca-2_{LLAMA_SIZE}b_type_5.csv"
    # print(type5_results(path5))
