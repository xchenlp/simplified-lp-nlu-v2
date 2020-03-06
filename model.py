from sklearn import preprocessing
from random import shuffle
import numpy as np
import collections

import tensorflow as tf
from keras.layers import Dense, Dropout, Activation, Conv1D, GlobalMaxPooling1D
from keras.models import Sequential, model_from_json
from keras import backend as K


class Model():
    def __init__(self, model_cfg):
        self.model = None
        self.le_encoder = None
        self.loaded = False
        self.session = None
        self.graph = None
        self.model_cfg = model_cfg

    def train(self, dataset):
        '''
        Train a model for a given dataset
        Dataset should be a list of tuples consisting of
        training sentence and the class label
        '''

        K.clear_session()
        graph = tf.Graph()
        with graph.as_default():
            session = tf.Session()
            with session.as_default():
                session.run(tf.global_variables_initializer())
                (x_train, y_train, le_encoder) = self.__preprocess(dataset)
                model = self.__build_model(num_classes=len(le_encoder.classes_))

                model.fit(x_train, y_train,
                          batch_size=cfg['model']['batch_size'],
                          epochs=cfg['model']['epochs'])
                log.info('finished training')
                ml_util.save(model, le_encoder, self.modelMeta)

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
        log.info('train %s intents with %s samples' % (len(set(labels)), len(data)))
        log.info(collections.Counter(labels))
        log.info(le_encoder.classes_)
        # vectorized_data = api.getEmbeddings(data)
        # vectorized_data = api.getEmbeddingsAsync(data)
        vectorized_data = api.getEmbeddingChunked(data)
        log.info(vectorized_data.shape)

        split_point = int(len(vectorized_data) * .9)
        x_train = vectorized_data  # vectorized_data[:split_point]
        y_train = encoded_labels  # encoded_labels[:split_point]

        x_train = ml_util.pad_trunc(x_train, cfg['model']['maxlen'])

        x_train = np.reshape(x_train, (len(x_train), cfg['model']['maxlen'], cfg['model']['embedding_dims']))
        y_train = np.array(y_train)
        return x_train, y_train, le_encoder

    def __build_model(self, num_classes=2, type='keras'):
        log.info('Build model')
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
        batch_size = model_cfg.get('batch_size', 32)
        drop_out = model_cfg.get('dropout', 0.2)
        strides = model_cfg.get('strides', 1)
        embedding_dims = model_cfg.get('embedding_dims', 300)
        filters = model_cfg.get('filters', 250)
        activation_type = model_cfg.get('activation', 'relu')
        kernel_size = model_cfg.get('kernel_size', 3)
        epochs = model_cfg.get('epochs', 10)
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
        # model.add(Dropout(drop_out))
        model.add(Activation(activation_type))
