import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, f1_score, average_precision_score, make_scorer
import umap

from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D
import multiprocessing

from optuna.integration import OptunaSearchCV

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

def word2vec_embed(sentences):
    w2v_size=300
    w2v_window=5
    w2v_min_count=1
    w2v_epochs=100
    maxlen = 24 # adapt to length of sentences
    sentences = [simple_preprocess(text) for text in sentences]

    w2v_model = Word2Vec(min_count=w2v_min_count, window=w2v_window,
                                                vector_size=w2v_size,
                                                seed=314,
                                                # workers=1)
                                               workers=multiprocessing.cpu_count())

    w2v_model.build_vocab(sentences)
    w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=w2v_epochs)

    model_vectors = w2v_model.wv
    w2v_words = model_vectors.index_to_key

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sentences)
    x_sentences = pad_sequences(tokenizer.texts_to_sequences(sentences),
                                                         maxlen=maxlen,
                                                         padding='post') 
                                                       
    num_words = len(tokenizer.word_index) + 1

    # weights matrix
    print("Create Embedding matrix ...")
    w2v_size = 300
    word_index = tokenizer.word_index
    vocab_size = len(word_index) + 1
    embedding_matrix = np.zeros((vocab_size, w2v_size))
    i=0
    j=0
        
    for word, idx in word_index.items():
        i +=1
        if word in w2v_words:
            j +=1
            embedding_vector = model_vectors[word]
            if embedding_vector is not None:
                embedding_matrix[idx] = model_vectors[word]
                
    word_rate = np.round(j/i,4)
    print("Word embedding rate : ", word_rate)
    print("Embedding matrix: %s" % str(embedding_matrix.shape))

    # Model creation

    input=Input(shape=(len(x_sentences),maxlen),dtype='float64')
    word_input=Input(shape=(maxlen,),dtype='float64')  
    word_embedding=Embedding(input_dim=vocab_size,
                             output_dim=w2v_size,
                             weights = [embedding_matrix],
                             input_length=maxlen)(word_input)
    word_vec=GlobalAveragePooling1D()(word_embedding)  
    embed_model = Model([word_input],word_vec)
    
    embed_model.summary()

    embeddings = embed_model.predict(x_sentences)
    print(embeddings.shape)

    return embeddings

def train_model_with_optuna(vectorizer, model, param_distributions, X_train, y_train, X_test, y_test, cv=5, n_trials=20, scoring="f1", random_state=314):

    # Define pipeline
    pipeline = Pipeline([
        ("vectorizer", vectorizer),
        ("classifier", model)
    ])
    
    # OptunaSearchCV for hyperparameter tuning
    search = OptunaSearchCV(
        pipeline,
        param_distributions=param_distributions,
        cv=cv,
        n_trials=n_trials,
        scoring=scoring,
        random_state=random_state,
        verbose=1,
        #n_jobs=-1,
    )
    
    # Fit the model
    search.fit(X_train, y_train)

    # Evaluate on test set
    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    #pr_auc = average_precision_score(y_test, y_pred)

    print(f"Best parameters: {search.best_params_}")
    print(f"Test Accuracy: {acc}")
    print(f"F1 Score: {f1}")
    print(classification_report(y_test, y_pred))
    
    return best_model, acc