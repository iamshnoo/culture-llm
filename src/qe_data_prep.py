import pandas as pd
import numpy as np
from process_text import is_translatable
import json

df1 = pd.read_csv("../data/alpacas/english.csv")
df2 = pd.read_csv("../data/alpacas/chinese.csv")
df3 = pd.read_csv("../data/alpacas/greek.csv")
df4 = pd.read_csv("../data/alpacas/hindi.csv")
df5 = pd.read_csv("../data/alpacas/persian.csv")
df6 = pd.read_csv("../data/alpacas/swahili.csv")


# filter out the rows that have nan values
overall_indices = []
for idx, row in df1.iterrows():
    text1 = row["input"]
    text2 = row["instruction"]
    text3 = row["output"]
    if text1 is np.nan or text2 is np.nan or text3 is np.nan:
        print("idx", idx, "has nan values")
        continue
    else:
        if (
            is_translatable(str(text1))
            and is_translatable(str(text2))
            and is_translatable(str(text3))
        ):
            overall_indices.append(idx)

# filter the other dataframes based on the indices
df1 = df1.iloc[overall_indices]
df2 = df2.iloc[overall_indices]
df3 = df3.iloc[overall_indices]
df4 = df4.iloc[overall_indices]
df5 = df5.iloc[overall_indices]
df6 = df6.iloc[overall_indices]

data = {}
for lang, df in zip(["zh", "el", "hi", "fa", "sw"], [df2, df3, df4, df5, df6]):
    inputs, instructions, outputs = [], [], []
    for a, b in zip(df1.iterrows(), df.iterrows()):
        inputs.append({"src": str(a[1]["input"]), "mt": str(b[1]["input"])})
        instructions.append(
            {"src": str(a[1]["instruction"]), "mt": str(b[1]["instruction"])}
        )
        outputs.append({"src": str(a[1]["output"]), "mt": str(b[1]["output"])})

    data[lang] = {"inputs": inputs, "instructions": instructions, "outputs": outputs}

with open(f"../data/alpacas/qe.json", "w") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)
