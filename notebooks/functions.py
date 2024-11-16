import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import umap

def pca_graph(X):
    pca = PCA()
    pca.fit_transform(X)

    sns.set_style('white')

    plt.figure(figsize=(8,6))
    fig, ax = plt.subplots()
    sns.lineplot(x=np.arange(pca.n_components_) + 1, y=pca.explained_variance_ratio_, color='blue', ax=ax)

    ax.set_xlabel('component')
    ax.set_ylabel('explained variance', color='blue')
    ax.yaxis.label.set_color('blue')
    ax.tick_params(axis='y', colors='blue')
    ax.spines['left'].set_color('blue')

    ax2 = ax.twinx()
    sns.lineplot(x=np.arange(pca.n_components_) + 1, y=pca.explained_variance_ratio_.cumsum(), color='red', ax=ax2)

    ax2.set_xlabel('component')
    ax2.set_ylabel('culumative explained variance', color='red')
    ax2.yaxis.label.set_color('red')
    ax2.tick_params(axis='y', colors='red')
    ax2.spines['left'].set_color('blue')
    ax2.spines['right'].set_color('red')

    plt.show()

def umap_plt(embed_X, y):
    plt.figure(figsize=(12, 12))
    ax = plt.gca()  # Get the current axis
    # Set background colors
    ax.set_facecolor('black')          # Axis background color
    plt.gcf().set_facecolor('black')    # Figure background color
    plt.scatter(embed_X[:, 0], embed_X[:, 1], s=.01, c=y.map({0: 'royalblue', 1: 'yellow'}))
    plt.title("UMAP Projection", color='white')
    ax.tick_params(colors='white')

    return plt

def umap_graph(embed_X, y):
    umap_plt(embed_X, y).show()