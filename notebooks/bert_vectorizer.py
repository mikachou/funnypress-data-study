from sklearn.base import BaseEstimator, TransformerMixin
from transformers import TFAutoModel, AutoTokenizer
import tensorflow as tf
import numpy as np
from tqdm import tqdm

class BertVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, model_name="camembert-base", batch_size=32):
        """
        Initialize the BertVectorizer with a specified model and batch size.

        Parameters:
        - model_name: str, name of the pretrained BERT model (default: 'camembert-base')
        - batch_size: int, batch size for processing (default: 32)
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.tokenizer = None
        self.model = None

    def _load_model(self):
        """Load the tokenizer and model."""
        if self.tokenizer is None or self.model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = TFAutoModel.from_pretrained(self.model_name)

    def _extract_embeddings(self, batch_sentences):
        """
        Extract embeddings for a batch of sentences.

        Parameters:
        - batch_sentences: List of str, sentences to process.

        Returns:
        - Tensor of shape (batch_size, hidden_size) containing sentence embeddings.
        """
        self._load_model()
        inputs = self.tokenizer(
            batch_sentences, return_tensors="tf", padding=True, truncation=True
        )
        outputs = self.model(**inputs)
        sentence_embeddings = tf.reduce_mean(outputs.last_hidden_state, axis=1)
        return sentence_embeddings

    def fit(self, X, y=None):
        """Fit does nothing as this transformer is stateless."""
        return self

    def transform(self, X):
        """
        Transform a list of sentences into their corresponding BERT embeddings.

        Parameters:
        - X: List of str, sentences to process.

        Returns:
        - np.ndarray of shape (len(X), hidden_size), containing embeddings.
        """
        self._load_model()

        all_embeddings = []
        num_batches = len(X) // self.batch_size + (len(X) % self.batch_size != 0)

        for i in tqdm(range(0, len(X), self.batch_size), total=num_batches, desc="Encoding Batches"):
            batch_sentences = X[i:i + self.batch_size]
            batch_embeddings = self._extract_embeddings(batch_sentences)
            all_embeddings.append(batch_embeddings)

        all_embeddings = tf.concat(all_embeddings, axis=0)
        return all_embeddings.numpy()