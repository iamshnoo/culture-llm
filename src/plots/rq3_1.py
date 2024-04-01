# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from math import pi

# Data for size 70 from the provided table
categories = [
    "  Beverage",
    "Female Clothing",
    "Female Names",
    "Food",
    "Literature",
    "Location",
    "Male Clothing",
    "Male Names",
    "Religion",
    "Overall",
]
values_size_70 = [
    [0.04, 0.57, 0.23, 0.38, 0.15, 0.15],
    [0.10, 0.39, 0.47, 0.11, 0.12, 0.26],
    [0.07, 0.71, 0.33, 0.39, 0.30, 0.37],
    [0.18, 0.43, 0.12, 0.53, 0.08, 0.35],
    [0.16, 0.28, 0.14, 0.65, 0.27, 0.08],
    [0.13, 0.39, 0.43, 0.49, 0.19, 0.26],
    [0.15, 0.73, 0.20, 0.06, 0.16, 0.19],
    [0.04, 0.76, 0.30, 0.35, 0.42, 0.34],
    [0.17, 0.50, 0.18, 0.16, 0.22, 0.19],
    [0.08, 0.39, 0.18, 0.26, 0.16, 0.14],
]

# Number of variables we're plotting
num_vars = len(categories)

# Compute angle for each axis in the plot (the plot is a circle so we need to divide the plot / number of variables)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

# The plot is circular, so we need to "complete the loop" and draw a line from the last value to the first.
values_size_70 += values_size_70[:1]
angles += angles[:1]

# Country names corresponding to each column in values_size_70
countries = ["USA", "China", "India", "Iran", "Kenya", "Greece"]

# Setting different colors for each country
cmap = "tab10"
colors = plt.get_cmap(cmap)(np.linspace(0, 1, len(countries)))

# Draw the radar chart for each country
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

# Draw one axe per variable and add labels
plt.xticks(angles[:-1], categories, fontsize=16)

# Draw ylabels
ax.set_rlabel_position(0)
plt.yticks(
    [0.2, 0.4, 0.6, 0.8],
    ["0.2", "0.4", "0.6", "0.8"],
    color="black",
    size=14,
    fontweight="bold",
)
plt.ylim(0, 1)

# Plot data for each country
for idx, country in enumerate(countries):
    country_values = [row[idx] for row in values_size_70[:-1]] + [
        values_size_70[0][idx]
    ]

    ax.plot(
        angles,
        country_values,
        linewidth=1,
        linestyle="solid",
        label=country,
        color=colors[idx],
    )

    # Fill area for each country
    ax.fill(angles, country_values, color=colors[idx], alpha=0.1)

    # set grid lines with dashed style for each angle
    ax.grid(True, color="#d3d3d3", linestyle="--")

    # Make all the circles even lighter and grey
    ax.spines["polar"].set_color("#d3d3d3")
    ax.spines["polar"].set_linewidth(0.3)

# Add legend
plt.legend(loc="upper right", bbox_to_anchor=(1.1, 1.1), fontsize=12)

# Show plot with legend
plt.show()

# Save plot
fig.savefig("../../figs/rq3_1_70.pdf", bbox_inches="tight", dpi=600)
