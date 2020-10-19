from model import tokenize_and_vectorize
from nltk.tokenize import TreebankWordTokenizer
import numpy as np
SEED = 7
np.random.seed(SEED)


def test_tokenize_and_vectorize():
    tokenizer = TreebankWordTokenizer()
    embedding_vector = {
        'test1': np.random.rand(300),
        'test2': np.random.rand(300)
    }
    dataset = [
        "test1 Test1 unknown",
        "test2 TEST2 UNknown"
    ]
    vecs = tokenize_and_vectorize(tokenizer, embedding_vector, dataset, 300)
    import pdb; pdb.set_trace()
