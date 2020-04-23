# Simplified LP NLU v2

This repo provides a simplified version of [LP NLU v2](https://lpgithub.dev.lprnd.net/BotCentral/cb_nlu_v2) repo, and a model coverted to Tensorflow 2.x native. To use it, simply do

```python
from model import Model  # for TF 1.x
# or, you can choose Tensorflow 2.x native model
from model_tf2 import Model  # for TF 2.x

model = Model(word2vec_pkl_path, config_path)
```

Find the `word2vec_pkl` file at `ca-gpu2:/data/cb_nlu_v2/vectors/wiki-news-300d-1M.pkl'`, and the `config` file in this repo (`config.yml`).

The `Model` class provides three functionalities: training, loading, and predicting. We document below the three functions. 

One can also see `demo.py` or `demo_tf2.py` for a demo on `ca-gpu2` server.

Remember to set `os.environ["CUDA_VISIBLE_DEVICES"] = '-1'`  to disable GPU. This is because on `ca-gpu2`, the cuda and cuDNN versions are wrong for TensorFlow (both 1.x and 2.x).

## Training

To train a model, simply provide the training set (json or csv format) path, and the model saving path

```python
model.train(tr_set_path, model_saving_path)
```

## Loading

To load a model, provide the model path

```python
model.load(model_path)
```

## Predicting

Prepare a list of messages (a list of `str` objects) and use the model's predict function:

```python
df_te = pandas.read_json(te_set_path, lines=True)
output = model.predict(list(df_te.text))
```

The output contains a list of dictionaries for each messages, recording the following fields of the prediction:

```python
'input', 'embeddings', 'label', 'highestProb', 'prob'
```



# Appendix: a design flaw in logit threshold code

In `model_tf2.py`, there’s one problem with line 125:

```python
self.get_logits = K.function([self.model.layers[0].input], [self.model.layers[4].output])
```

it only works when layers is set to 1 in the `Model.__build_graph` function. To collect the Keras logit scores, we seize the output of layer 4 of the deep learning model. However, only when `layers == 1`, the layer 4 output is the logit threshold score.

This is not too problematic now, because in production we always fix `layers` to 1. To fix this error. Consider if we can change layers[4] to layers[-2] (the penultimate layer of the network, which should always be the logit score layer, independent of the `layers` value).