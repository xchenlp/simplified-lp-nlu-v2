from model import tokenize_and_vectorize, Model
from nltk.tokenize import TreebankWordTokenizer
import numpy as np
import pytest
SEED = 7
np.random.seed(SEED)

@pytest.fixture
def model():
    return Model(encoder_type='transformer', gpu_id=-1)



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



def test_preprocess(model):
    data = ['This is a test data 1', 'This is test data']
    x_train, y_train, label_encoder = model.__preprocess(data)
    import pdb; pdb.set_trace()
    assert x_train.shape == (2, 400, 768)