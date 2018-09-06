import gensim
import multiprocessing
import numpy as np

class Word2Vec:

    def __init__(self, inputs, embedding_dim):
        self.inputs = inputs
        self.embedding_dim = embedding_dim
        self.window = 2
        self.skip_gram = 1
        self.min_count = 1
        self.num_workers = multiprocessing.cpu_count()

    def embed(self, model, elem):
        """If the word is not in embedding, return the zero vector."""
        try:
            return model.wv.get_vector(elem)
        except:
            return np.zeros([self.embedding_dim], dtype=np.float32)

    def get_embedding(self):
        model = gensim.models.Word2Vec(
            self.inputs,
            size=self.embedding_dim,
            window=self.window,
            sg=self.skip_gram,
            min_count=self.min_count,
            workers=self.num_workers
        )

        embeddings = [
            np.array([self.embed(model, elem) for elem in sequence] for sequence in self.inputs)
        ]

        return np.array(embeddings)
