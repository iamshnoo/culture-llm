import json
import random

import numpy as np
import pandas as pd
from tqdm import tqdm
import itertools

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

CATEGORIES = ["female_names", "male_names", "food", "female_clothing", "male_clothing", "location", "literature", "beverage", "religion"]

def load_data(prompts_path, targets_path):
    with open(prompts_path) as f:
        prompts = json.load(f)

    with open(targets_path) as f:
        targets = json.load(f)

    prompts = pd.DataFrame(prompts)
    prompts.reset_index(level=0, inplace=True)
    prompts = prompts.rename(columns={"index": "category"})
    prompts = prompts.explode("prompts")
    prompts.reset_index(drop=True, inplace=True)

    # Create DataFrame from the list of dictionaries
    targets = pd.DataFrame(targets)
    targets.reset_index(level=0, inplace=True)
    targets = targets.rename(columns={"index": "category"})

    merged_df = pd.merge(prompts, targets, on="category")
    return merged_df


def transform_df_type1(df, pre_prompt_template):
    # Create a new DataFrame with the desired columns
    transformed_df = pd.DataFrame(columns=['country', 'category', 'prompt', 'correct', 'incorrect', 'options'])

    all_countries = df.columns[2:]  # Update if the column structure is different

    # Iterate over each row of the original DataFrame
    for index, row in tqdm(df.iterrows(), total=len(df)):
        for country in all_countries:  # Assuming first two columns are 'category' and 'prompts'
            correct_options = row[country]

            # Create a question for each correct option
            for correct_option in correct_options:
                new_row = {}
                new_row['country'] = country
                new_row['category'] = row['category']
                new_row['prompt'] = pre_prompt_template.format(country=country) + row['prompts']
                new_row['correct'] = correct_option

                incorrect_options_set = set()
                for other_country in df.columns[2:]:
                    if other_country != country:
                        other_options = row[other_country]
                        if other_options:
                            sampled_option = random.choice(other_options)
                            if sampled_option not in incorrect_options_set:
                                incorrect_options_set.add(sampled_option)

                incorrect_options = list(incorrect_options_set)
                new_row['incorrect'] = incorrect_options

                # Combine and shuffle options
                options = [correct_option] + incorrect_options
                random.shuffle(options)
                new_row['options'] = options

                # Append the new row to the transformed DataFrame
                transformed_df = pd.concat([transformed_df, pd.DataFrame([new_row])])

    return transformed_df


def transform_df_type2(df, pre_prompt_template):
    # Create a new DataFrame with the desired columns
    transformed_df = pd.DataFrame(columns=['country', 'category', 'prompt', 'correct', 'incorrect', 'options'])

    # List of all countries
    all_countries = df.columns[2:]  # Update if the column structure is different

    # Iterate over each row of the original DataFrame
    for index, row in tqdm(df.iterrows(), total=len(df)):
        for country in all_countries:
            # Determine number of countries to select
            num_countries_to_select = len(all_countries) - 1
            assert num_countries_to_select > 0, "There should be at least two countries"

            # Select other countries randomly, with one repeated to reach the target number
            other_countries = [c for c in all_countries if c != country]
            selected_countries = random.sample(other_countries, min(num_countries_to_select, len(other_countries)))
            while len(selected_countries) < num_countries_to_select:
                selected_countries.append(random.choice(other_countries))

            # Generate 50 unique sets of options
            option_combinations = set()
            while len(option_combinations) < 50:
                option_set = tuple(random.choice(row[sc]) for sc in selected_countries if row[sc])
                if option_set:
                    option_combinations.add(option_set)

            # Create a question for each combination of options
            for combination in option_combinations:
                new_row = {}
                new_row['country'] = country
                new_row['category'] = row['category']
                new_row['prompt'] = pre_prompt_template.format(country=country) + row['prompts']
                new_row['correct'] = None
                new_row['incorrect'] = list(combination)

                # Combine and shuffle options
                options = list(combination)
                random.shuffle(options)
                new_row['options'] = options

                # Append the new row to the transformed DataFrame
                transformed_df = pd.concat([transformed_df, pd.DataFrame([new_row])])

    return transformed_df


def transform_df_type3(df, pre_prompt_template):
    # Create a new DataFrame with the desired columns
    transformed_df = pd.DataFrame(columns=['country', 'category', 'prompt', 'correct', 'incorrect', 'options'])

    # List of all countries
    all_countries = df.columns[2:]  # Update if the column structure is different

    # Iterate over each row of the original DataFrame
    for index, row in tqdm(df.iterrows(), total=len(df)):
        for country in all_countries:
            # Select multiple correct options from the correct country
            correct_options = row[country]

            # Generate 50 unique sets of 3 correct options
            unique_correct_sets = set()
            while len(unique_correct_sets) < 50 and len(correct_options) >= 3:
                selected_correct_options = tuple(random.sample(correct_options, 3))
                unique_correct_sets.add(selected_correct_options)

            for correct_set in unique_correct_sets:
                # Create a new row for the transformed DataFrame
                new_row = {}

                # Set country and category
                new_row['country'] = country
                new_row['category'] = row['category']

                # Modify the prompt as needed and add to new_row
                new_row['prompt'] = pre_prompt_template.format(country=country) + row['prompts']

                # Select one incorrect option from a different country
                other_countries = [c for c in all_countries if c != country]
                incorrect_country = random.choice(other_countries)
                incorrect_options = row[incorrect_country]
                selected_incorrect_option = [random.choice(incorrect_options)] if incorrect_options else []

                # Combine and shuffle options
                options = list(correct_set) + selected_incorrect_option
                random.shuffle(options)
                new_row['options'] = options

                new_row['correct'] = list(correct_set)
                new_row['incorrect'] = selected_incorrect_option

                # Append the new row to the transformed DataFrame
                transformed_df = pd.concat([transformed_df, pd.DataFrame([new_row])])

    return transformed_df


def transform_df_type4(df, pre_prompt_template):
    # Create a new DataFrame with the desired columns
    transformed_df = pd.DataFrame(columns=['country', 'category', 'prompt', 'correct', 'incorrect', 'options'])

    # List of all countries
    all_countries = df.columns[2:]  # Update if the column structure is different

    # Iterate over each row of the original DataFrame
    for index, row in tqdm(df.iterrows(), total=len(df)):
        for country in all_countries:
            # Select multiple correct options from the correct category and country
            correct_options = row[country]

            # Generate up to 50 unique sets of correct options, or the number of unique correct options available
            unique_correct_sets = set()
            n = min(50, len(set(correct_options)))
            while len(unique_correct_sets) < n:
                selected_correct_option = tuple([random.choice(correct_options)])
                unique_correct_sets.add(selected_correct_option)

            for correct_set in unique_correct_sets:
                # Create a new row for the transformed DataFrame
                new_row = {}

                # Set country and category
                new_row['country'] = country
                new_row['category'] = row['category']

                # Modify the prompt as needed and add to new_row
                new_row['prompt'] = pre_prompt_template.format(country=country) + row['prompts']

                # Set correct option
                new_row['correct'] = list(correct_set)[0]

                # Select incorrect options from different categories within the same country
                incorrect_options_set = set()
                allowed_categories = [c for c in CATEGORIES if c != row['category']]
                rand_categories = random.sample(allowed_categories, 3)
                assert row['category'] not in rand_categories, "The correct category should not be in the list of random categories"

                for rand_category in rand_categories:
                    # filter the df to get the rows with the rand_category
                    rand_category_df = df[df.category == rand_category]
                    # choose a random row from the rand_category_df
                    rand_row = rand_category_df.sample(n=1)
                    # get the incorrect options from the rand_row
                    incorrect_options = rand_row[country].values
                    # sample one incorrect option from the incorrect_options
                    incorrect_options = random.sample(list(incorrect_options), 1)[0]
                    # add the incorrect options to the incorrect_options_set
                    incorrect_options_set.update(incorrect_options)


                # for other_row in df.itertuples():
                #     if other_row.category != row['category'] and len(incorrect_options_set) < 3:
                #         other_options = getattr(other_row, country)
                #         if other_options:
                #             sampled_option = random.choice(other_options)
                #             if sampled_option not in incorrect_options_set:
                #                 incorrect_options_set.add(sampled_option)

                incorrect_options = list(incorrect_options_set)

                # Trim the incorrect options if necessary
                incorrect_options = incorrect_options[:3]

                new_row['incorrect'] = incorrect_options

                # Combine and shuffle options
                options = [new_row['correct']] + incorrect_options
                random.shuffle(options)
                new_row['options'] = options

                # Append the new row to the transformed DataFrame
                transformed_df = pd.concat([transformed_df, pd.DataFrame([new_row])])

    return transformed_df


def transform_df_type5(df, pre_prompt_template):
    # Create a new DataFrame with the desired columns
    transformed_df = pd.DataFrame(columns=['country', 'category', 'prompt', 'correct', 'incorrect', 'options'])

    # List of all countries and categories
    all_countries = df.columns[2:]
    all_categories = ['female_clothing', 'female_names', 'male_clothing', 'male_names']
    paired_categories = {'female_clothing': 'male_clothing', 'male_clothing': 'female_clothing',
                        'female_names': 'male_names', 'male_names': 'female_names'}

    # Filter out the categories that are not in all_categories
    df = df[df.category.isin(all_categories)]

    # Iterate over each row of the original DataFrame
    for index, row in tqdm(df.iterrows(), total=len(df)):
        for country in all_countries:
            # Set country and category
            category = row['category']
            paired_category = paired_categories[category]

            # Get correct and incorrect options
            correct_options = list(itertools.chain.from_iterable(df.loc[df['category'] == category, c].values[0] for c in all_countries if c != country))
            incorrect_options = df.loc[df['category'] == paired_category, country].values[0]

            # Generate unique sets of options
            max_combinations = min(50, len(correct_options)//2, len(incorrect_options)//2)
            unique_combinations = set()
            while len(unique_combinations) < max_combinations:
                selected_corrects = tuple(random.sample(correct_options, 2))
                selected_incorrects = tuple(random.sample(incorrect_options, 2))
                combination = (selected_corrects, selected_incorrects)
                unique_combinations.add(combination)

            # Create new rows for each combination
            for corrects, incorrects in unique_combinations:
                new_row = {}
                new_row['country'] = country
                new_row['category'] = category
                new_row['prompt'] = pre_prompt_template.format(country=country) + row['prompts']
                new_row['correct'] = list(corrects)
                new_row['incorrect'] = list(incorrects)
                options = list(corrects) + list(incorrects)
                random.shuffle(options)
                new_row['options'] = options
                transformed_df = pd.concat([transformed_df, pd.DataFrame([new_row])])

    return transformed_df


def transform_df(df, df_type, pre_prompt_template, replace_mask=False, post_template=""):
    if df_type in [1, "type1"]:
        transformed_df = transform_df_type1(df, pre_prompt_template)
    elif df_type in [2, "type2"]:
        transformed_df = transform_df_type2(df, pre_prompt_template)
    elif df_type in [3, "type3"]:
        transformed_df = transform_df_type3(df, pre_prompt_template)
    elif df_type in [4, "type4"]:
        transformed_df = transform_df_type4(df, pre_prompt_template)
    elif df_type in [5, "type5"]:
        transformed_df = transform_df_type5(df, pre_prompt_template)

    if replace_mask:
        transformed_df['prompt'] = transformed_df['prompt'].apply(lambda x: x.replace("[mask]", "[BLANK]"))

    if post_template:
        transformed_df['prompt'] = transformed_df['prompt'].apply(lambda x: x + post_template)

    return transformed_df


if __name__ == "__main__":
    PRE_PROMPT = "This scenario is situated in {country}. "
    TYPE = "type1"
    prompts_path = "../data/camel/prompts.json"
    targets_path = "../data/camel/targets.json"

    merged_df = load_data(prompts_path, targets_path)
    transformed_merged_df = transform_df(merged_df, TYPE, PRE_PROMPT, replace_mask=True, post_template=" What is the best option to fill in the [BLANK] ?")

    # Save the transformed DataFrame
    print(transformed_merged_df.head())
    print(len(transformed_merged_df))
