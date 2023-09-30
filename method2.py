import math

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from umap import UMAP
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


VAR_X = 140
VAR_Y = 35


def reduced_cloud_points(dataset):
    fig, axes = plt.subplots(1, 3)

    for label in dataset:
        # PCA
        pca = PCA(n_components=2)
        data = pca.fit_transform(dataset[label])

        axes[0].scatter(data[:, 0], data[:, 1], label=label)
        axes[0].set_title("ACP")

        # TSNE
        tsne = TSNE(n_components=2)
        data = tsne.fit_transform(dataset[label])

        axes[1].scatter(data[:, 0], data[:, 1], label=label)
        axes[1].set_title("TSNE")

        # UMAP
        umap = UMAP(n_components=2)
        data = umap.fit_transform(dataset[label])

        axes[2].scatter(data[:, 0], data[:, 1], label=label)
        axes[2].set_title("UMAP")
    
    plt.legend()

    plt.show()


def joint_histograms(dataset):
    fig, axes = plt.subplots(5, 4)

    def hist(label_x, label_y, row, col):
        x = dataset[label_x][:, VAR_X]
        y = dataset[label_y][:, VAR_X]

        # Oversampling
        if len(x) < len(y):
            x = np.append(x, np.random.choice(x, size=len(y) - len(x)))
        elif len(y) < len(x):
            y = np.append(y, np.random.choice(y, size=len(x) - len(y)))
        
        ax = axes[row, col]
        ax.hist2d(x, y, bins=50)
        ax.set_xlabel(label_x)
        ax.set_ylabel(label_y)

        x = dataset[label_x][:, VAR_Y]
        y = dataset[label_y][:, VAR_Y]

        if len(x) < len(y):
            x = np.append(x, np.random.choice(x, size=len(y) - len(x)))
        elif len(y) < len(x):
            y = np.append(y, np.random.choice(y, size=len(x) - len(y)))
        
        ax = axes[row, col + 2]
        ax.hist2d(x, y, bins=30)
        ax.set_xlabel(label_x)
        ax.set_ylabel(label_y)

    hist("PRAD", "BRCA", 0, 0)
    hist("PRAD", "KIRC", 0, 1)
    hist("PRAD", "LUAD", 1, 0)
    hist("PRAD", "COAD", 1, 1)
    hist("BRCA", "KIRC", 2, 0)
    hist("BRCA", "LUAD", 2, 1)
    hist("BRCA", "COAD", 3, 0)
    hist("KIRC", "LUAD", 3, 1)
    hist("KIRC", "COAD", 4, 0)
    hist("LUAD", "COAD", 4, 1)

    plt.subplots_adjust(left=None, right=None, top=None, bottom=None, wspace=None, hspace=None)

    plt.show()


def cloud_points(dataset):
    for label in dataset:
        x = dataset[label][:, VAR_X]
        y = dataset[label][:, VAR_Y]

        plt.scatter(x, y, label=label)
    
    plt.xlabel("Gène {}".format(VAR_X))
    plt.ylabel("Gène {}".format(VAR_Y))
    plt.title("Nuage de points")

    plt.legend()

    plt.show()


def class_histogram(dataset):
    fig, axes = plt.subplots(1, 2)

    def hist(var, col):
        for label in dataset:
            data = dataset[label][:, var]
            axes[col].hist(data, label=label, bins=30, alpha=0.4)
        
        axes[col].set_xlabel("Gène {}".format(var))

    hist(VAR_X, 0)
    hist(VAR_Y, 1)

    plt.tight_layout()
    plt.legend()

    plt.show()


def load_dataset():
    data = pd.read_csv("data.csv").values
    labels = pd.read_csv("labels.csv").values

    dataset = {}

    for sample, label in zip(data[:,1:], labels[:,1]):
        if label not in dataset:
            dataset[label] = []
        dataset[label].append(sample)
    
    # Conversion to 2D np.array
    for label in dataset:
        dataset[label] = np.array(dataset[label])

    return dataset


def main():
    dataset = load_dataset()

    class_histogram(dataset)
    cloud_points(dataset)
    joint_histograms(dataset)
    reduced_cloud_points(dataset)


if __name__ == "__main__":
    main()
