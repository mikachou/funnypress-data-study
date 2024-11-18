from sklearn.base import BaseEstimator, TransformerMixin
from gensim.utils import simple_preprocess
from gensim.models import Word2Vec
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D
import numpy as np
import multiprocessing

class Word2VecVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, w2v_size=300, w2v_window=5, w2v_min_count=1, w2v_epochs=100, maxlen=24):
        self.w2v_size = w2v_size
        self.w2v_window = w2v_window
        self.w2v_min_count = w2v_min_count
        self.w2v_epochs = w2v_epochs
        self.maxlen = maxlen
        self.tokenizer = None
        self.embedding_matrix = None
        self.embed_model = None

    def fit(self, X, y=None):
        # Preprocess and tokenize sentences
        sentences = [simple_preprocess(text) for text in X]

        # Train Word2Vec model
        self.w2v_model = Word2Vec(
            sentences=sentences,
            vector_size=self.w2v_size,
            window=self.w2v_window,
            min_count=self.w2v_min_count,
            workers=multiprocessing.cpu_count(),
            seed=314
        )
        self.w2v_model.build_vocab(sentences)
        self.w2v_model.train(
            sentences, 
            total_examples=self.w2v_model.corpus_count, 
            epochs=self.w2v_epochs
        )

        # Create tokenizer and pad sequences
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(sentences)
        x_sentences = pad_sequences(
            self.tokenizer.texts_to_sequences(sentences),
            maxlen=self.maxlen,
            padding='post'
        )

        # Build the embedding matrix
        vocab_size = len(self.tokenizer.word_index) + 1
        self.embedding_matrix = np.zeros((vocab_size, self.w2v_size))
        for word, idx in self.tokenizer.word_index.items():
            if word in self.w2v_model.wv:
                self.embedding_matrix[idx] = self.w2v_model.wv[word]

        # Build the embedding model
        word_input = Input(shape=(self.maxlen,), dtype='float64')
        word_embedding = Embedding(
            input_dim=vocab_size,
            output_dim=self.w2v_size,
            weights=[self.embedding_matrix],
            input_length=self.maxlen,
            trainable=False
        )(word_input)
        word_vec = GlobalAveragePooling1D()(word_embedding)
        self.embed_model = Model(word_input, word_vec)

        return self

    def transform(self, X, y=None):
        # Preprocess and tokenize sentences
        sentences = [simple_preprocess(text) for text in X]
        x_sentences = pad_sequences(
            self.tokenizer.texts_to_sequences(sentences),
            maxlen=self.maxlen,
            padding='post'
        )
        # Generate embeddings using the embedding model
        embeddings = self.embed_model.predict(x_sentences, verbose=0)
        return embeddings
