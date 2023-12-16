import pandas as pd
from datasets import load_dataset

replace_dict = {
        "en" : {
            " in the United States": "",
            " in the Untied States": "",
            "In United States, ": "",
            "American " : "",
            " in United States": "",
            " of the United States": "",
            " the United States" : " the country",
            " in America" : "",
            " an American" : " a",
            " Americans" : " people",
        },
        "hi": {
            "अमेरिका में": "",
            "अमेरिकी": "",
            "अमेरिका के": "देश के",
            "अमेरिका": "देश",
            "अमेरिकी लोगों": "लोगों",
            "अमेरिकी": "",
        },
        "zh": {
            "在美国": "",
            "美国人": "人们",
            "美国的": "该",
            "美国": "该国",
            "美国人的": "人们的",
            "美国大城市": "大城市",
            "美国大部分地区": "该国大部分地区",
            "美国传统": ""
        },
        "sw": {
            "nchini Marekani": "",
            "Wamarekani": "Watu",
            "Waamerikani": "Watu",
            "wamarekani": "watu",
            "Marekani": "nchi",
            "Nchini Marekani": "Nchini",
            "Nchini marekani": "Nchini",
            "nchini marekani": "nchini",
            "Kimarekani": "Kienyeji",
        },
        "fa": {
            "در ایالت متحده آمریکا": "",
            "آمریکایی": "",
            "آمریکایی ها": "مردم",
            "ایالت متحده آمریکا": "کشور",
            "در آمریکا": "",
            "آمریکا": "کشور",
        },
        "el": {
            "στις Ηνωμένες Πολιτείες": "",
            "Αμερικανοί": "άνθρωποι",
            "στην Αμερική": "εδώ",
            "Ηνωμένες Πολιτείες": "χώρα",
            "Ηνωμένες Πολιτίεες": "χώρα",
            "Αμερικανική": "",
            "αμερικανική": "",
        },
}

def get_no_country_questions(lang):
    dataset = load_dataset("iamshnoo/geomlama", split=f"{lang}")
    df = pd.DataFrame(dataset)

    # extract every 6th row, starting from the 0th row
    df = df.iloc[::6, :]
    for key, value in replace_dict[f"{lang}"].items():
        df["question"] = df["question"].str.replace(key, value)

    df["question"] = df["question"].str.capitalize().str.strip().str.replace("，", "")

    return df

if __name__ == "__main__":
    LANG = "el"
    d = get_no_country_questions(LANG)
    print(d["question"].values)
