from sklearn import preprocessing
import random
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
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import losses, optimizers
from early_stopping import EarlyStoppingAtMaxMacroF1
import json
import hashlib

from encoder import Encoder

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
                np.random.seed(int(hashlib.sha1(token.encode()).hexdigest(), 16) % (10 ** 6))
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


def save(model, le, path, history):
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
    np.save(labels_file, le.categories_[0])
    with open(os.path.join(path, "log.json"), 'w') as f:
        json.dump(history.history, f)


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
        #le = preprocessing.LabelEncoder()
        categories = np.load(labels_file)
        le = preprocessing.OneHotEncoder(handle_unknown='ignore', sparse=False)
        le.fit([[c] for c in categories])
        json_file.close()
        return model, le


def predict(session, graph, model, vectorized_input, num_classes):
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
            preds = to_categorical(preds, num_classes=num_classes)
            return (probs, preds)


def createPermutationDataSet(dataset=[], entities=[]):
    entity_dict = {
        d['entity_name'] : d['entity_values'] for d in entities
    }
    print('[createPermutationDataSet] Pre: [dataset size]: %d [num of entity]: %d' % (len(dataset), len(entities)))
    print('[createPermutationDataSet] Entity Counter: [%s]' % ', '.join(['%s:%d' %(k,len(v)) for k,v in entity_dict.items()]))
    permutationDS = collections.defaultdict(lambda: [])
    resultDS = []
    for idx, d in enumerate(dataset):
        data = d['data']
        label = d['label']
        for ENT_NAME in entity_dict.keys():
            if ENT_NAME in data:
                for val in entity_dict[ENT_NAME]:
                    new_data = data.replace(ENT_NAME, val)
                    permutationDS[label].append(new_data)
        resultDS.append({'data': data, 'label': label})
    for label, values in permutationDS.items():
        sample_values = values
        if len(values) > 100:
            sample_values = random.sample(values, 100)

        [resultDS.append({'data': v, 'label': label}) for v in sample_values]
    print('[createPermutationDataSet] Post: [dataset size]: %d ' % (len(resultDS),))
    return resultDS

class Model:
    def __init__(self, word2vec_pkl_path=None, config_path='./config.yml', label_smoothing=0, encoder_type='fasttext', gpu_id=-1):
        with open(config_path, 'r') as f:
            self.model_cfg = yaml.safe_load(f)['model']
        if encoder_type == 'fasttext':
            self.tokenizer = TreebankWordTokenizer()

            with open(word2vec_pkl_path, 'rb') as f:
                self.vectors = pickle.load(f)
        self.model = None
        self.session = None
        self.graph = None
        self.le_encoder = None # label encoder
        self.encoder = None
        self.label_smoothing = label_smoothing
        self.encoder_type = encoder_type
        self.gpu_id = gpu_id

    def train(self, tr_set_path: str, save_path: str, entities_path: str=None,  va_split: float=0.1, stratified_split: bool=False, early_stopping: bool=True):
        """
        Train a model for a given dataset
        Dataset should be a list of tuples consisting of
        training sentence and the class label
        Args:
            tr_set_path: path to training data
            save_path: path to save model weights and labels
            va_split: fraction of training data to be used for validation in early stopping. Only effective when stratified_split is set to False. Will be overridden if stratified_split is True. 
            stratified_split: whether to split training data stratified by class. If True, validation will be done on a fixed val set from a stratified split out of the training set with the fraction of va_split. 
            early_stopping: whether to do early stopping
        Returns: 
            history of training including average loss for each training epoch
            
        """
        df_tr = read_csv_json(tr_set_path)
        entities = None if entities_path is None else self.__get_entities(entities_path)
        if stratified_split:
            df_va = df_tr.groupby('intent').apply(lambda g: g.sample(frac=va_split, random_state=SEED))
            df_tr = df_tr[~df_tr.index.isin(df_va.index.get_level_values(1))]
            va_messages, va_labels = list(df_va.text), list(df_va.intent)
            va_dataset = [{'data': va_messages[i], 'label': va_labels[i]} for i in range(len(df_va))]
            tr_messages, tr_labels = list(df_tr.text), list(df_tr.intent)
            tr_dataset = [{'data': tr_messages[i], 'label': tr_labels[i]} for i in range(len(df_tr))]
            (x_train, y_train, le_encoder) = self.__preprocess(tr_dataset) if entities is None else \
                                            self.__preprocess(tr_dataset, entities)
            (x_va, y_va, _) = self.__preprocess(va_dataset, le_encoder=le_encoder) if entities is None else \
                                            self.__preprocess(va_dataset, entities, le_encoder)
        else:
            tr_messages, tr_labels = list(df_tr.text), list(df_tr.intent)
            tr_dataset = [{'data': tr_messages[i], 'label': tr_labels[i]} for i in range(len(df_tr))]
            (x_train, y_train, le_encoder) = self.__preprocess(tr_dataset) if entities is None else \
                                            self.__preprocess(tr_dataset, entities)

        K.clear_session()
        graph = tf.Graph()
        with graph.as_default():
            session = tf.Session()
            with session.as_default():
                session.run(tf.global_variables_initializer())
                model = self.__build_model(num_classes=len(le_encoder.categories_[0]))
                model.compile(
                      loss=losses.CategoricalCrossentropy(label_smoothing=self.label_smoothing),
                      #metrics=['categorical_accuracy'],
                      optimizer=self.model_cfg.get('optimizer', 'adam') #default lr at 0.001
                      #optimizer=optimizers.Adam(learning_rate=5e-4)
                )
                # early stopping callback using validation loss 
                callback = tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    min_delta=0,
                    patience=5,
                    verbose=0,
                    mode="auto",
                    baseline=None,
                    restore_best_weights=True,
                )
                #callback = EarlyStoppingAtMaxMacroF1(
                #    patience=100, # record all epochs
                #    validation=(x_va, y_va)
                #)

                print('start training')
                history = model.fit(x_train, y_train,
                          batch_size=self.model_cfg['batch_size'],
                          epochs=100,
                          validation_split=va_split if not stratified_split else 0,
                          validation_data=(x_va, y_va) if stratified_split else None,
                          callbacks=[callback] if early_stopping else None)
                history.history['train_data'] = tr_set_path
                print(f'finished training in {len(history.history["loss"])} epochs')
                save(model, le_encoder, save_path, history)
                self.model = model
                self.session = session
                self.graph = graph
                self.le_encoder = le_encoder
                # return training history 
                return history.history
    

    def __get_entities(entities_path: str):
        with open(entities_path) as f:
            entities = json.load(f)
        return entities


    def __preprocess(self, dataset, entities=None, le_encoder=None):
        '''
        Preprocess the dataset, transform the categorical labels into numbers.
        Get word embeddings for the training data.
        '''
        if entities:
            dataset = createPermutationDataSet(dataset, entities)
        random.shuffle(dataset)
        data = [s['data'] for s in dataset]
        #labels = [s['label'] for s in dataset]
        labels = [[s['label']] for s in dataset]
        #le_encoder = preprocessing.LabelEncoder()
        if le_encoder is None: 
            le_encoder = preprocessing.OneHotEncoder(handle_unknown='ignore', sparse=False)
            le_encoder.fit(labels)
        encoded_labels = le_encoder.transform(labels)
        print('%s intents with %s samples' % (len(le_encoder.get_feature_names()), len(data)))
        #print('train %s intents with %s samples' % (len(set(labels)), len(data)))
        #print(collections.Counter(labels))
        print(le_encoder.categories_[0])

        y_train = encoded_labels  # encoded_labels[:split_point]
        y_train = np.array(y_train)

        # tokenize and encode
        if self.encoder_type == 'fasttext':
            vectorized_data = tokenize_and_vectorize(self.tokenizer, self.vectors, data, self.model_cfg['embedding_dims'])

            # split_point = int(len(vectorized_data) * .9)
            x_train = vectorized_data  # vectorized_data[:split_point]
            x_train = pad_trunc(x_train, self.model_cfg['maxlen'])
            x_train = np.reshape(x_train, (len(x_train), self.model_cfg['maxlen'], self.model_cfg['embedding_dims']))
            
        elif self.encoder_type == 'transformer':
            encoder = Encoder('distilmbert', max_sent_len=self.model_cfg['embedding_dims'], pad_to_max_sent_len=True, gpu_id=self.gpu_id)
            x_train = encoder(data)
            # convert to numpy array for Keras classifier
            x_train = x_train.cpu().detach().numpy()

        else:
            raise NotImplementedError("encoder type not recognized")

        return x_train, y_train, le_encoder


    def __build_model(self, num_classes=2, type='keras'):
        print('Build model')
        model = Sequential()
        layers = self.model_cfg.get('layers', 1)
        for l in range(layers):
            self.__addLayers(model, self.model_cfg)
        model.add(Dense(num_classes))
        model.add(Activation('softmax'))

        return model

    def __addLayers(self, model, model_cfg):
        maxlen = model_cfg.get('maxlen', 400)
        strides = model_cfg.get('strides', 1)
        embedding_dims = model_cfg.get('embedding_dims', 300) if self.encoder_type == 'fasttext' else 768 # DistilBERT
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
        if self.encoder_type == 'fasttext':
            vectorized_data = tokenize_and_vectorize(self.tokenizer, self.vectors, input, self.model_cfg['embedding_dims'])
            x_train = pad_trunc(vectorized_data, self.model_cfg['maxlen'])
            vectorized_input = np.reshape(x_train, (len(x_train), self.model_cfg['maxlen'], self.model_cfg['embedding_dims']))
        elif self.encoder_type == 'transformer':
            encoder = Encoder('distilmbert', max_sent_len=self.model_cfg['embedding_dims'], gpu_id=self.gpu_id)
            x_train = encoder(input)
            # convert to numpy array for Keras classifier
            x_train = x_train.cpu().detach().numpy()

        (probs, preds) = predict(self.session, self.graph, self.model, vectorized_input, len(self.le_encoder.categories_[0]))
        probs = probs.tolist()
        results = self.le_encoder.inverse_transform(preds)
        output = [{'input': input[i],
                   'embeddings': x_train[i],
                   #'label': r,
                   'label': r.item(),
                   'highestProb': max(probs[i]),
                   #'prob': dict(zip(self.le_encoder.classes_, probs[i]))
                   'prob': dict(zip(self.le_encoder.categories_[0], probs[i]))
                   } for i, r in enumerate(results)]
        return output
