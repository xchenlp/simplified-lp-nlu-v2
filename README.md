# Simplified LP NLU v2

This repo provides a simplified version of [LP NLU v2](https://lpgithub.dev.lprnd.net/BotCentral/cb_nlu_v2) repo, with *label smoothing* cross entopy. It's weird that in this repo they are using label smoothing, while in their production they don't (thus the repo linked above should not be CB's production repo).

To use it, simply do

```python
from model import Model

model = Model(word2vec_pkl_path, config_path)
```

Find the `word2vec_pkl` file at `ca-gpu2:/data/cb_nlu_v2/vectors/wiki-news-300d-1M.pkl'`, and the `config` file in this repo (`config.yml`).

The `Model` class provides three functionalities: training, loading, and predicting. We document below the three functions. 

One can also see `demo.py` for a demo on `ca-gpu2` server. Remember to set `os.environ["CUDA_VISIBLE_DEVICES"] = '-1'`  to disable GPU. This is because on `ca-gpu2`, the cuDNN version is wrong for tensorflow.

## training

To train a model, simply provide the training set (json or csv format) path, and the model saving path

```python
model.train(tr_set_path, model_saving_path)
```

## loading

To load a model, provide the model path

```python
model.load(model_path)
```

## predicting

Prepare a list of messages (a list of `str` objects) and use the model's predict function:

```python
df_te = pandas.read_json(te_set_path, lines=True)
output = model.predict(list(df_te.text))
```

The output contains a list of dictionaries for each messages, recording the following fields of the prediction:

```python
'input', 'embeddings', 'label', 'highestProb', 'prob'
```

