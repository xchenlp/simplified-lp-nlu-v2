import torch.nn
import time
import os
import numpy
import logging
from typing import List
import json

logger = logging.getLogger(__name__)


class Fasttext(torch.nn.Module):
    r"""
    Fasttext encoder that maps text sentence to word embeddings to be used in classifiers. There are several ways to
    load weights: either using the FastText txt format of vector embeddings, or using the Pytorch's load function
    to load the Pytorch pickled weights. The latter is faster.

    If load the FastText txt format weights, one can choose whether to save the PyTorch pickled version to save time
    when loading the same embedding next time.

    Args:
        options_file: the option files for FastText
        pickle_model_path: the path to the Pytorch pickled weights. This is the preferred way of loading, because the file is smaller in size and the loading time is much faster
        text_model_path: the path to the FastText txt model. This is required if and only if pickle_model_path is not provided
        pickle_model_save_path: When using a FastText txt model, provide this path if one wants to save the pytorch pickled version of the weights, so one can use the Pytorch pickle loading for the next time

    Shape:
        embeddings: :math:`(batch_size, sentence_length)` a list of list of str (tokenized words)
        Output: :math:`(batch_size, sentence_length, self.embedding_size)` word embeddings
    """
    def __init__(self, options_file, pickle_model_path=None, text_model_path=None, pickle_model_save_path=None):
        super().__init__()

        with open(options_file) as json_file:
            options = json.load(json_file)
        self.embedding_size = options['hidden_size']

        if pickle_model_path is not None:
            self._load_pickle_model(pickle_model_path)
        else:
            assert text_model_path is not None
            self._load_text_model(text_model_path, pickle_model_save_path)

        self.word2vec['|||PAD|||'] = numpy.zeros((self.embedding_size,), dtype=numpy.float32)  # assign padded tokens zero weights
        self.output_device = torch.device('cpu')

    def _load_pickle_model(self, pickle_model_path):
        logger.info(f'Loading Fasttext embeddings from a pickle model at {pickle_model_path}')
        self.word2vec = torch.load(pickle_model_path)

    def _load_text_model(self, text_model_path, pickle_model_save_path):
        logger.info(f'Loading Fasttext embeddings from a text model at {text_model_path}')
        self.word2vec = {}
        with open(text_model_path, 'rt') as f:
            for line in f:
                vf = []
                word, *vec = line.rstrip().split()
                for v in vec:
                    vf.append(float(v))
                if len(vf) == self.embedding_size:
                    self.word2vec[word] = numpy.array(vf, dtype=numpy.float32)
                else:
                    logger.info(f'Found word {word} not conform to the embedding length. the embedding is {vf}. This should only filter out the first line of the vec file for fasttext')
        logger.info(f'Finished loading {len(self.word2vec)} words.')
        if pickle_model_save_path is not None:
            logger.info(f'save pickle model at {pickle_model_save_path}')
            torch.save(self.word2vec, pickle_model_save_path)

    def forward(self, list_of_tokenized_messages: List[List[str]]) -> torch.Tensor:

        minibatch_size = len(list_of_tokenized_messages)
        sentence_length = max([len(x) for x in list_of_tokenized_messages])
        message_tensor = numpy.zeros((minibatch_size, sentence_length, self.embedding_size), dtype=numpy.float32)
        for i, tokenized_message in enumerate(list_of_tokenized_messages):
            for j, token in enumerate(tokenized_message):
                if token in self.word2vec:
                    weights = self.word2vec[token]
                else:  # randomly embed UNKNOWN tokens
                    # https://medium.com/@martinpella/how-to-use-pre-trained-word-embeddings-in-pytorch-71ca59249f76
                    weights = numpy.random.normal(scale=0.6, size=(self.embedding_size,))
                message_tensor[i, j, :] = weights

        return torch.tensor(message_tensor).to(self.output_device)
