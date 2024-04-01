import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_style("whitegrid")
sns.set_context("paper")

og_data = {
    "Prompt": [
        "USA",
        "USA",
        "USA",
        "China",
        "China",
        "China",
        "India",
        "India",
        "India",
        "Iran",
        "Iran",
        "Iran",
        "Kenya",
        "Kenya",
        "Kenya",
        "Greece",
        "Greece",
        "Greece",
    ],
    "Size": [7, 13, 70, 7, 13, 70, 7, 13, 70, 7, 13, 70, 7, 13, 70, 7, 13, 70],
    "USA": [
        0.0,
        0.0,
        0.0,
        0.1,
        0.09,
        0.08,
        0.07,
        0.07,
        0.06,
        0.09,
        0.08,
        0.07,
        0.08,
        0.08,
        0.07,
        0.08,
        0.07,
        0.06,
    ],
    "China": [
        0.36,
        0.33,
        0.36,
        0.0,
        0.0,
        0.0,
        0.37,
        0.33,
        0.35,
        0.40,
        0.37,
        0.40,
        0.37,
        0.34,
        0.34,
        0.37,
        0.33,
        0.36,
    ],
    "India": [
        0.17,
        0.18,
        0.16,
        0.24,
        0.25,
        0.23,
        0.0,
        0.0,
        0.0,
        0.21,
        0.24,
        0.22,
        0.19,
        0.21,
        0.20,
        0.18,
        0.20,
        0.18,
    ],
    "Iran": [
        0.22,
        0.22,
        0.23,
        0.27,
        0.29,
        0.32,
        0.24,
        0.27,
        0.29,
        0.0,
        0.0,
        0.0,
        0.23,
        0.25,
        0.27,
        0.22,
        0.23,
        0.26,
    ],
    "Kenya": [
        0.14,
        0.15,
        0.13,
        0.22,
        0.21,
        0.18,
        0.18,
        0.19,
        0.17,
        0.16,
        0.17,
        0.15,
        0.0,
        0.0,
        0.0,
        0.15,
        0.16,
        0.14,
    ],
    "Greece": [
        0.12,
        0.12,
        0.12,
        0.17,
        0.16,
        0.18,
        0.14,
        0.13,
        0.13,
        0.14,
        0.14,
        0.16,
        0.13,
        0.13,
        0.13,
        0.0,
        0.0,
        0.0,
    ],
}

df_original = pd.DataFrame(og_data)

# Extracting data for size 70
df_size_70_manual = (
    df_original[df_original["Size"] == 70].drop("Size", axis=1).set_index("Prompt")
)

# Data preparation
data_size_70 = df_size_70_manual.reset_index()
data_size_70_melted = data_size_70.melt(
    id_vars="Prompt", var_name="Target Country", value_name="Value"
)

# Preparing data for a 100% stacked bar chart
stacked_data = df_size_70_manual.apply(lambda x: x / x.sum(), axis=1)

# Plotting a 100% Stacked Bar Chart
cmap = "tab10"
ax = stacked_data.plot(
    kind="bar", stacked=True, colormap=cmap, figsize=(10, 8), edgecolor="black"
)

plt.xlabel("Prompt Country", fontsize=30, labelpad=15)
plt.ylabel("Percentage (%)", fontsize=30, labelpad=15)
plt.xticks(rotation=0, fontsize=20)
plt.yticks(fontsize=28)
plt.legend(
    title="Countries",
    bbox_to_anchor=(1.05, 1),
    loc="upper left",
    fontsize=28,
    title_fontsize=28,
)
ax.yaxis.grid(True)  # Adding horizontal gridlines for better readability
ax.set_axisbelow(True)  # Ensure gridlines are behind the bars

plt.tight_layout()
plt.show()
ax.figure.savefig("../../figs/rq3_2_70.pdf", bbox_inches="tight", dpi=600)
