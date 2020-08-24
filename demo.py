from model import Model
import os
from pdb import set_trace
import pandas
from sklearn.metrics import classification_report


os.environ["CUDA_VISIBLE_DEVICES"] = '-1'  # disable gpu. This is because on ca-gpu2, the cuDNN version is wrong for tensorflow


def main():
    model = Model(word2vec_pkl_path='/data/cb_nlu_v2/vectors/wiki-news-300d-1M.pkl', config_path='config.yml')
    vertical = 'finserv'
    test_model_path = f'/data/cb_nlu_{vertical}_cross_entropy_loss_label_smoothing/e_0.02'
    tr_set_path = f'/data/starter_pack_datasets/{vertical}/tr.json'
    te_set_path = f'/data/starter_pack_datasets/{vertical}/te.json'
    print("start training")
    ######################### training #########################
    model.train(tr_set_path, test_model_path)
    ######################### training ends #########################
    print("start testing")
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
    threshold_predictions = [x['label'] if x['highestProb'] >= 0.6 else 'undefined' for x in output]
    df_te.loc[:, 'pred_intent'] = threshold_predictions
    df_te.loc[:, 'pred_score'] = [x['highestProb'] for x in output]
    df_te.loc[:, 'prob'] = [x['prob'] for x in output]
    df_te.to_json(f'/data/cb_nlu_results/{vertical}/te_preds_xentroy_smoothing_0.02.json', orient='records', lines=True)
    print(classification_report(y_true=ground_truths, y_pred=threshold_predictions))
    ######################### evaluating the prediction ends #########################
    

if __name__ == '__main__':
    main()
