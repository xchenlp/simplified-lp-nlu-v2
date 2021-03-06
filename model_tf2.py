from sklearn import preprocessing
from random import shuffle
import numpy as np
import collections

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Activation, Conv1D, GlobalMaxPooling1D
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras import backend as K
from gensim.models.keyedvectors import KeyedVectors
from nltk.tokenize import TreebankWordTokenizer
import re
import pickle
from scipy.special import softmax
import os

import yaml
import pandas
from keras.models import Model as KerasModel
from typing import List


def read_csv_json(file_name) -> pandas.DataFrame:
    if file_name.endswith('json') or file_name.endswith('jsonl'):
        df = pandas.read_json(file_name, lines=True)
    elif file_name.endswith('csv'):
        df = pandas.read_csv(file_name)
    else:
        raise NotImplementedError
    return df


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
                # print('token not found: (%s) in sentence: %s' % (token, ' '.join(tokens)))
                if unk_vec is not None:
                    vecs.append(unk_vec)
                continue
        vectorized_data.append(vecs)
    return vectorized_data


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


class Model:
    def __init__(self, word2vec_pkl_path, config_path):
        with open(config_path, 'r') as f:
            self.model_cfg = yaml.safe_load(f)['model']
        self.tokenizer = TreebankWordTokenizer()

        with open(word2vec_pkl_path, 'rb') as f:
            self.vectors = pickle.load(f)
        self.model = None
        # self.logits_model = None

        self.le_encoder = None
        self.labels = None
        self.get_logits = None

    def train(self, tr_set_path, save_path):
        """
        Train a model for a given dataset
        Dataset should be a list of tuples consisting of
        training sentence and the class label
        """
        df_tr = read_csv_json(tr_set_path)
        messages = list(df_tr.text)
        labels = list(df_tr.intent)
        dataset = [{'data': messages[i], 'label': labels[i]} for i in range(len(df_tr))]

        (x_train, y_train, le_encoder) = self.__preprocess(dataset)
        model = self.__build_model(num_classes=len(le_encoder.classes_))

        print('start training')
        model.fit(x_train, y_train,
                    batch_size=self.model_cfg['batch_size'],
                    epochs=self.model_cfg['epochs'])
        print('finished training')
        self.model = model
        self.get_logits = K.function([self.model.layers[0].input], [self.model.layers[4].output])
        self.le_encoder = le_encoder
        self.labels = self.le_encoder.classes_
        self.save(save_path)
        

        # self.logits_model = KerasModel(inputs=self.model.input, outputs=self.model.get_layer('logits').output)

    def save(self, path):
        '''
        save model based on model, encoder
        '''

        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        print(f'saving model to {path}')
        structure_file = os.path.join(path, 'structure.json')
        weight_file = os.path.join(path, 'weight.h5')
        labels_file = os.path.join(path, 'classes')
        with open(structure_file, "w") as json_file:
            json_file.write(self.model.to_json())
        self.model.save_weights(weight_file)
        np.save(labels_file, self.le_encoder.classes_)

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

        # split_point = int(len(vectorized_data) * .9)
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
        model.add(Dense(num_classes, name='logits'))
        model.add(Activation('softmax'))

        def categorical_crossentropy_w_label_smoothing(y_true, y_pred, from_logits=False, label_smoothing=0):
            y_pred = K.constant(y_pred) if not K.is_tensor(y_pred) else y_pred
            y_true = K.cast(y_true, y_pred.dtype)

            if label_smoothing is not 0:
                smoothing = K.cast_to_floatx(label_smoothing)

                def _smooth_labels():
                    num_classes = K.cast(K.shape(y_true)[1], y_pred.dtype)
                    return y_true * (1.0 - smoothing) + (smoothing / num_classes)

                y_true = K.switch(K.greater(smoothing, 0), _smooth_labels, lambda: y_true)
            return K.categorical_crossentropy(y_true, y_pred, from_logits=from_logits)

        # model.compile(loss=categorical_crossentropy_w_label_smoothing,
        #               metrics=['sparse_categorical_accuracy'],
        #               optimizer=optimizer_type)
        model.compile(loss='sparse_categorical_crossentropy',
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

    def load(self, path):
        print(f'loading model from {path}')
        structure_file = os.path.join(path, 'structure.json')
        weight_file = os.path.join(path, 'weight.h5')
        labels_file = os.path.join(path, 'classes.npy')
        with open(structure_file, "r") as json_file:
            json_string = json_file.read()

        self.model = model_from_json(json_string)
        self.model.load_weights(weight_file)
        self.model._make_predict_function()
        self.get_logits = K.function([self.model.layers[0].input], [self.model.layers[4].output])

        self.le_encoder = preprocessing.LabelEncoder()
        self.le_encoder.classes_ = np.load(labels_file)
        self.labels = self.le_encoder.classes_
        # self.logits_model = KerasModel(inputs=self.model.input, outputs=self.model.get_layer('logits').output)

    def predict(self, input: List[str]):
        vectorized_data = tokenize_and_vectorize(self.tokenizer, self.vectors, input)
        x_train = pad_trunc(vectorized_data, self.model_cfg['maxlen'])
        vectorized_input = np.reshape(x_train, (len(x_train), self.model_cfg['maxlen'], self.model_cfg['embedding_dims']))

        probs = self.model.predict_proba(vectorized_input)
        preds = self.model.predict_classes(vectorized_input)
        probs = probs.tolist()
        results = self.le_encoder.inverse_transform(preds)
        output = [{'input': input[i],
                   'embeddings': x_train[i],
                   'label': r if r is not None else 'undefined',
                   'highestProb': max(probs[i]),
                   'prob': dict(zip(self.le_encoder.classes_, probs[i]))
                   } for i, r in enumerate(results)]
        return output

    def encode(self, list_of_messages):
        vectorized_data = tokenize_and_vectorize(self.tokenizer, self.vectors, list_of_messages)
        x_train = pad_trunc(vectorized_data, self.model_cfg['maxlen'])
        vectorized_input = np.reshape(x_train, (len(x_train), self.model_cfg['maxlen'], self.model_cfg['embedding_dims']))
        return vectorized_input

    def predict_threshold(self, list_of_messages, undefined_logit_score=2, other_label_name='undefined', encoded=False):
        assert other_label_name not in self.labels  # the labels list should not contain the other label 
        if not encoded:
            vectorized_input = self.encode(list_of_messages)
        else:
            vectorized_input = list_of_messages

        logit_scores = self.get_logits([vectorized_input])[0]

        other_tuple = (other_label_name, undefined_logit_score)

        pred_labels = []
        sorted_confs = []
        for i in range(logit_scores.shape[0]):
            label_conf_tuples = list(zip(self.labels, logit_scores[i]))
            label_conf_tuples.append(other_tuple)
            label_conf_tuples.sort(key=lambda x: x[1], reverse=True)

            new_pred_labels_i = [x[0] for x in label_conf_tuples]

            new_sorted_confs_i = [x[1] for x in label_conf_tuples]
            # apply softmax
            new_sorted_confs_i = softmax(new_sorted_confs_i)

            # push into the lists
            pred_labels.append(new_pred_labels_i)
            sorted_confs.append(new_sorted_confs_i)
        return pred_labels, sorted_confs

    # def get_logits(self, input, logits_model):
    #     x_train = self.tokenize_and_vectorize(input)
    #     logits = logits_model.predict(x_train)
    #     return logits
