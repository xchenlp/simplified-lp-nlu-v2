from sklearn import preprocessing
from random import shuffle
import numpy as np
import collections

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.keras.layers import Dense, Dropout, Activation, Conv1D, GlobalMaxPooling1D
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras import backend as K
from gensim.models.keyedvectors import KeyedVectors
from nltk.tokenize import TreebankWordTokenizer
import re
import pickle
import os

import yaml
import pandas
from typing import List
from tensorflow.keras import losses

SEED = 7

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


def tokenize_and_vectorize(tokenizer, embedding_vector, dataset, embedding_dims):
    vectorized_data = []
    # probably could be optimized further
    ds1 = [use_only_alphanumeric(samp.lower()) for samp in dataset]
    token_list = [tokenizer.tokenize(sample) for sample in ds1]

    for tokens in token_list:
        vecs = []
        for token in tokens:
            try:
                vecs.append(embedding_vector[token].tolist())
            except KeyError:
                # print('token not found: (%s) in sentence: %s' % (token, ' '.join(tokens)))
                np.random.seed(hash(token) % 1000000)
                unk_vec = np.random.rand(embedding_dims)
                vecs.append(unk_vec.tolist())
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


def save(model, le, path):
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
        json_file.write(model.to_json())
    model.save_weights(weight_file)
    np.save(labels_file, le.classes_)


def load(path):
    print(f'loading model from {path}')
    structure_file = os.path.join(path, 'structure.json')
    weight_file = os.path.join(path, 'weight.h5')
    labels_file = os.path.join(path, 'classes.npy')
    with open(structure_file, "r") as json_file:
        json_string = json_file.read()
        model = model_from_json(json_string)
        model.load_weights(weight_file)
        model._make_predict_function()
        le = preprocessing.LabelEncoder()
        le.classes_ = np.load(labels_file)
        json_file.close()
        return model, le


def predict(session, graph, model, vectorized_input):
    if session is None:
        raise ("Session is not initialized")
    if graph is None:
        raise ("Graph is not initialized")
    if model is None:
        raise ("Model is not initialized")
    with session.as_default():
        with graph.as_default():
            probs = model.predict_proba(vectorized_input)
            preds = model.predict_classes(vectorized_input)
            return (probs, preds)


class Model:
    def __init__(self, word2vec_pkl_path, config_path):
        with open(config_path, 'r') as f:
            self.model_cfg = yaml.safe_load(f)['model']
        self.tokenizer = TreebankWordTokenizer()

        with open(word2vec_pkl_path, 'rb') as f:
            self.vectors = pickle.load(f)
        self.model = None
        self.session = None
        self.graph = None
        self.le_encoder = None

    def train(self, tr_set_path, save_path, va_split=0.1, stratified_split=False):
        """
        Train a model for a given dataset
        Dataset should be a list of tuples consisting of
        training sentence and the class label
        """
        df_tr = read_csv_json(tr_set_path)
        if stratified_split:
            df_va = df_tr.groupby('intent').apply(lambda g: g.sample(frac=va_split, random_state=SEED))
            df_tr = df_tr[~df_tr.index.isin(df_va.index.get_level_values(1))]
        
        tr_messages, va_messages = list(df_tr.text), list(df_va.text)
        tr_labels, va_labels = list(df_tr.intent), list(df_va.intent)
        tr_dataset = [{'data': tr_messages[i], 'label': tr_labels[i]} for i in range(len(df_tr))]
        va_dataset = [{'data': va_messages[i], 'label': va_labels[i]} for i in range(len(df_va))]

        K.clear_session()
        graph = tf.Graph()
        with graph.as_default():
            session = tf.Session()
            with session.as_default():
                session.run(tf.global_variables_initializer())
                (x_train, y_train, le_encoder) = self.__preprocess(tr_dataset)
                (x_va, y_va, _) = self.__preprocess(va_dataset)
                model = self.__build_model(num_classes=len(le_encoder.classes_))

                callback = tf.keras.callbacks.EarlyStopping(
                    monitor="val_sparse_categorical_accuracy",
                    min_delta=0,
                    patience=5,
                    verbose=0,
                    mode="auto",
                    baseline=None,
                    restore_best_weights=True,
                )

                print('start training')
                history = model.fit(x_train, y_train,
                          batch_size=self.model_cfg['batch_size'],
                          epochs=100,
                          validation_split=va_split,
                          validation_data=(x_va, y_va) if stratified_split else None,
                          callbacks=[callback])
                print(f'finished training in {len(history.history["loss"])} epochs')
                save(model, le_encoder, save_path)
                self.model = model
                self.session = session
                self.graph = graph
                self.le_encoder = le_encoder

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
        vectorized_data = tokenize_and_vectorize(self.tokenizer, self.vectors, data, self.model_cfg['embedding_dims'])

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
        model.add(Dense(num_classes))
        model.add(Activation('softmax'))

        def categorical_crossentropy_w_label_smoothing(y_true, y_pred,
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

        # model.compile(loss=categorical_crossentropy_w_label_smoothing,
        #               metrics=['sparse_categorical_accuracy'],
        #               optimizer=optimizer_type)
        custom_metrics = Metrics()
        model.compile(loss='sparse_categorical_crossentropy',
                      metrics=['sparse_categorical_accuracy', custom_metrics],
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
        K.clear_session()
        graph = tf.Graph()
        with graph.as_default():
            session = tf.Session()
            with session.as_default():
                self.session = session
                self.graph = graph
                (model, le) = load(path)
                self.model = model
                self.le_encoder = le

    def predict(self, input: List[str]):
        vectorized_data = tokenize_and_vectorize(self.tokenizer, self.vectors, input, self.model_cfg['embedding_dims'])
        x_train = pad_trunc(vectorized_data, self.model_cfg['maxlen'])
        vectorized_input = np.reshape(x_train, (len(x_train), self.model_cfg['maxlen'], self.model_cfg['embedding_dims']))

        (probs, preds) = predict(self.session, self.graph, self.model, vectorized_input)
        probs = probs.tolist()
        results = self.le_encoder.inverse_transform(preds)
        output = [{'input': input[i],
                   'embeddings': x_train[i],
                   'label': r,
                   'highestProb': max(probs[i]),
                   'prob': dict(zip(self.le_encoder.classes_, probs[i]))
                   } for i, r in enumerate(results)]
        return output
