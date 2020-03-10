from gensim.models.keyedvectors import KeyedVectors
from nltk.tokenize import TreebankWordTokenizer
import re
import pickle
import os
from pdb import set_trace


def use_only_alphanumeric(input):
    pattern = re.compile('[\W^\'\"]+')
    output = pattern.sub(' ', input).strip()
    return output


def tokenize_and_vectorize(tokenizer, embedding_vector, dataset):
    vectorized_data = []
    # probably could be optimized further
    ds1 = [use_only_alphanumeric(samp) for samp in dataset]
    token_list = [tokenizer.tokenize(sample) for sample in ds1]

    unk_vec = None
    try:
        unk_vec = embedding_vector['UNK'].tolist()
    except Exception as e:
        pass

    for tokens in token_list:
        vecs = []
        for token in tokens:
            try:
                vecs.append(embedding_vector[token].tolist())
            except KeyError:
                print('token not found: (%s) in sentence: %s' % (token, ' '.join(tokens)))
                if unk_vec is not None:
                    vecs.append(unk_vec)
                continue
        vectorized_data.append(vecs)
    return vectorized_data


def main():
    print('load tokenizer')
    tokenizer = TreebankWordTokenizer()

    word2vec_path = '/data/cb_nlu_v2/vectors/wiki-news-300d-1M.vec'
    word2vec_pkl_path = '/data/cb_nlu_v2/vectors/wiki-news-300d-1M.pkl'
    if not os.path.isfile(word2vec_pkl_path):
        print('load vectors')
        vectors = KeyedVectors.load_word2vec_format(word2vec_path, limit=None)
        vectors.init_sims(replace=True)  # normalize the word vectors
        with open(word2vec_pkl_path, 'wb') as f:
            pickle.dump(vectors, f)
    else:
        print('load pickle vectors')
        with open(word2vec_pkl_path, 'rb') as f:
            vectors = pickle.load(f)

    set_trace()

    sentences = ['hi , how are you typosss', 'hello']
    tokenize_and_vectorize(tokenizer, vectors, sentences)
    set_trace()


if __name__ == '__main__':
    main()
