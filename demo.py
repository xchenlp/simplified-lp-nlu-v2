from model import Model
import os
from pdb import set_trace
import pandas
from sklearn.metrics import classification_report


os.environ["CUDA_VISIBLE_DEVICES"] = '-1'  # disable gpu. This is because on ca-gpu2, the cuDNN version is wrong for tensorflow


def main():
    model = Model(word2vec_pkl_path='/data/cb_nlu_v2/vectors/wiki-news-300d-1M.pkl', config_path='config.yml')

    test_model_path = '/data/cb_nlu_test_model'
    tr_set_path = '/data/deep-sentence-classifiers/preprocessed_data/Hawaiian/tr.json'
    te_set_path = '/data/deep-sentence-classifiers/preprocessed_data/Hawaiian/te.json'

    ######################### training #########################
    model.train(tr_set_path, test_model_path)
    ######################### training ends #########################


    ######################### loading #########################
    model.load(test_model_path)
    ######################### loading ends #########################


    ######################### predicting #########################
    df_te = pandas.read_json(te_set_path, lines=True)
    output = model.predict(list(df_te.text))
    ######################### predicting ends #########################


    ######################### evaluating the prediction #########################
    ground_truths = list(df_te.intent)
    predictions = [x['label'] for x in output]
    print(classification_report(y_true=ground_truths, y_pred=predictions))

    ######################### see the prediction dictionary fields #########################
    print(list(output[0].keys()))


if __name__ == '__main__':
    main()
