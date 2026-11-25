"""Make results/accuracy.svg from results/classification.csv."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main():
    df = pd.read_csv("results/classification.csv")
    piv = df.pivot(index="dataset", columns="method", values="accuracy")
    piv = piv[["1NN-Euclidean", "1NN-DTW (full)", "1NN-DTW (band 10%)", "MiniRocket-500"]]
    ax = piv.plot.bar(figsize=(9, 4.5), width=0.8, colormap="viridis")
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(0.7, 1.02)
    ax.set_title("UCR test accuracy (official train/test split)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, fontsize=8, frameon=False)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("results/accuracy.svg")


if __name__ == "__main__":
    main()
