import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import umap
import umap.plot

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
