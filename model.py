from sklearn import preprocessing
from random import shuffle
import numpy as np
import collections

import tensorflow as tf
from keras.layers import Dense, Dropout, Activation, Conv1D, GlobalMaxPooling1D
from keras.models import Sequential, model_from_json
from keras import backend as K
from gensim.models.keyedvectors import KeyedVectors
from nltk.tokenize import TreebankWordTokenizer
import re
import pickle
import os
from pdb import set_trace
from embeddings import tokenize_and_vectorize
import yaml
from jians_utils import read_csv_json


def pad_trunc(data, maxlen):
    """
    For a given dataset pad with zero vectors or truncate to maxlen
    """
    new_data = []
    # Create a vector of 0s the length of our word vectors
    zero_vector = []
    for _ in range(len(data[0][0])):
        zero_vector.append(0.0)

    for sample in data:
        if len(sample) > maxlen:
            temp = sample[:maxlen]
        elif len(sample) < maxlen:
            temp = list(sample)
            # Append the appropriate number 0 vectors to the list
            additional_elems = maxlen - len(sample)
            for _ in range(additional_elems):
                temp.append(zero_vector)
        else:
            temp = sample
        new_data.append(temp)
    return new_data


def save(model, le):
    '''
    save model based on model, encoder
    '''

    fileMeta = ('/data/cb_nlu_v2/', '1', '1')
    if not os.path.exists('%s%s/%s' % fileMeta):
        os.makedirs('%s%s/%s' % fileMeta, exist_ok=True)
    print('saving model:::(DIR:%s, ModelID:%s, ModelVersion:%s)' % fileMeta)
    structure_file = "%s%s/%s/structure.json" % fileMeta
    weight_file = "%s%s/%s/weight.h5" % fileMeta
    labels_file = "%s%s/%s/classes" % fileMeta
    with open(structure_file, "w") as json_file:
        json_file.write(model.to_json())
    model.save_weights(weight_file)
    np.save(labels_file, le.classes_)
    print('finished save model:::(DIR:%s, ModelID:%s, ModelVersion:%s)' % fileMeta)


class Model:
    def __init__(self):
        self.model_cfg = self.load_cfg()
        self.tokenizer = TreebankWordTokenizer()
        self.vectors = self._load_vectors

    @staticmethod
    def _load_vectors():
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
        return vectors

    @staticmethod
    def load_cfg():
        with open("config.yml", 'r') as f:
            return yaml.safe_load(f)['model']

    def train(self, dataset):
        """
        Train a model for a given dataset
        Dataset should be a list of tuples consisting of
        training sentence and the class label
        """

        K.clear_session()
        graph = tf.Graph()
        with graph.as_default():
            session = tf.Session()
            with session.as_default():
                session.run(tf.global_variables_initializer())
                (x_train, y_train, le_encoder) = self.__preprocess(dataset)
                model = self.__build_model(num_classes=len(le_encoder.classes_))

                print('start training')
                model.fit(x_train, y_train,
                          batch_size=self.model_cfg['batch_size'],
                          epochs=self.model_cfg['epochs'])
                print('finished training')
                save(model, le_encoder)

    def __preprocess(self, dataset):
        '''
        Preprocess the dataset, transform the categorical labels into numbers.
        Get word embeddings for the training data.
        '''
        shuffle(dataset)
        data = [s['data'] for s in dataset]
        labels = [s['label'] for s in dataset]
        le_encoder = preprocessing.LabelEncoder()
        le_encoder.fit(labels)
        encoded_labels = le_encoder.transform(labels)
        print('train %s intents with %s samples' % (len(set(labels)), len(data)))
        print(collections.Counter(labels))
        print(le_encoder.classes_)
        vectorized_data = tokenize_and_vectorize(self.tokenizer, self.vectors, data)

        print(vectorized_data.shape)

        split_point = int(len(vectorized_data) * .9)
        x_train = vectorized_data  # vectorized_data[:split_point]
        y_train = encoded_labels  # encoded_labels[:split_point]

        x_train = pad_trunc(x_train, self.model_cfg['maxlen'])

        x_train = np.reshape(x_train, (len(x_train), self.model_cfg['maxlen'], self.model_cfg['embedding_dims']))
        y_train = np.array(y_train)
        return x_train, y_train, le_encoder

    def __build_model(self, num_classes=2, type='keras'):
        print('Build model')
        model = Sequential()
        optimizer_type = self.model_cfg.get('optimizer', 'adam')
        layers = self.model_cfg.get('layers', 1)
        for l in range(layers):
            self.__addLayers(model, self.model_cfg)
        model.add(Dense(num_classes))
        model.add(Activation('softmax'))

        def categorical_crossentropy_w_label_smoothing(y_true, y_pred, \
                                                       from_logits=False, label_smoothing=0):
            y_pred = K.constant(y_pred) if not K.is_tensor(y_pred) else y_pred
            y_true = K.cast(y_true, y_pred.dtype)

            if label_smoothing is not 0:
                smoothing = K.cast_to_floatx(label_smoothing)

                def _smooth_labels():
                    num_classes = K.cast(K.shape(y_true)[1], y_pred.dtype)
                    return y_true * (1.0 - smoothing) + (smoothing / num_classes)

                y_true = K.switch(K.greater(smoothing, 0), _smooth_labels, lambda: y_true)
            return K.categorical_crossentropy(y_true, y_pred, from_logits=from_logits)

        model.compile(loss=categorical_crossentropy_w_label_smoothing,
                      metrics=['sparse_categorical_accuracy'],
                      optimizer=optimizer_type)
        return model

    def __addLayers(self, model, model_cfg):
        maxlen = model_cfg.get('maxlen', 400)
        strides = model_cfg.get('strides', 1)
        embedding_dims = model_cfg.get('embedding_dims', 300)
        filters = model_cfg.get('filters', 250)
        activation_type = model_cfg.get('activation', 'relu')
        kernel_size = model_cfg.get('kernel_size', 3)
        hidden_dims = model_cfg.get('hidden_dims', 200)

        model.add(Conv1D(
            filters,
            kernel_size,
            padding='valid',
            activation=activation_type,
            strides=strides,
            input_shape=(maxlen, embedding_dims)))
        model.add(GlobalMaxPooling1D())
        model.add(Dense(hidden_dims))
        model.add(Activation(activation_type))


def main():
    model = Model()

    # prepare data from our json training file
    df_tr = read_csv_json('/data/deep-sentence-classifiers/preprocessed_data/Hawaiian/tr.json')

    messages = list(df_tr.text)
    labels = list(df_tr.intent)
    dataset = [{'data': messages[i], 'label': labels[i]} for i in range(len(df_tr))]

    # train
    model.train(dataset)

    # # predict
    # model.predict(...)


if __name__ == '__main__':
    main()
