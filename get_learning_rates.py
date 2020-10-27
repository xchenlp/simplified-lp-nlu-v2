from model import Model
import os
import pandas
from sklearn.metrics import classification_report
import json


os.environ["CUDA_VISIBLE_DEVICES"] = '-1'  # disable gpu. This is because on ca-gpu2, the cuDNN version is wrong for tensorflow

"""
balanced sets: 
5 x 20, 40  
9 x 20, 100, 200
20 x 20, 50
32 x 20, 50
cross vertical full set
"""
def main():
    model = Model(word2vec_pkl_path='/data/cb_nlu_v2/vectors/wiki-news-300d-1M.pkl', config_path='config.yml', label_smoothing=0.02)

    test_model_path = '/data/cb_nlu_test_model_label_smoothing'
    tr_set_paths = [
        '/data/deep-sentence-classifiers/preprocessed_data/Hawaiian/tr_5_intents_20_messages_per_intent.json',
        '/data/deep-sentence-classifiers/preprocessed_data/Hawaiian/tr_5_intents_40_messages_per_intent.json',
        '/data/starter_pack_datasets/telco/tr_20_per_class.json',
        '/data/starter_pack_datasets/telco/tr_100_per_class.json',
        '/data/starter_pack_datasets/telco/tr_200_per_class.json',
        '/data/starter_pack_datasets/cross_vertical/tr_20_intents_20_messages_per_intent.json',
        '/data/starter_pack_datasets/cross_vertical/tr_20_intents_50_messages_per_intent.json',
        '/data/starter_pack_datasets/cross_vertical/tr_20_per_class.json',
        '/data/starter_pack_datasets/cross_vertical/tr_50_per_class.json',
        '/data/starter_pack_datasets/cross_vertical/tr.json'
    ]
    # te_set_paths = [
    #     '/data/deep-sentence-classifiers/preprocessed_data/Hawaiian/te.json',
    #     '/data/deep-sentence-classifiers/preprocessed_data/Hawaiian/te.json',
    #     '/data/starter_pack_datasets/telco/te.json',
    #     '/data/starter_pack_datasets/telco/te.json',
    #     '/data/starter_pack_datasets/telco/te.json',
    #     '/data/starter_pack_datasets/cross_vertical/te.json',
    #     '/data/starter_pack_datasets/cross_vertical/te.json',
    #     '/data/starter_pack_datasets/cross_vertical/te.json',
    #     '/data/starter_pack_datasets/cross_vertical/te.json',
    #     '/data/starter_pack_datasets/cross_vertical/te.json'
    # ]
    # assert len(tr_set_paths) == len(te_set_paths)
    histories = []
    for tr_set_path in tr_set_paths:

        ######################### training #########################
        history = model.train(tr_set_path, test_model_path, stratified_split=True, early_stopping=False)
        histories.append(history)

        with open(os.path.join(test_model_path, 'summary_logs.json'), 'w') as f:
            json.dump(histories, f)
        ######################### training ends #########################

        ######################### loading #########################
        # model.load(test_model_path)
        # ######################### loading ends #########################

        # ######################### predicting #########################
        # df_te = pandas.read_json(te_set_path, lines=True)
        # output = model.predict(list(df_te.text))
        # ######################### predicting ends #########################

        # ######################### evaluating the prediction #########################
        # ground_truths = list(df_te.intent)
        # predictions = [x['label'] for x in output]
        # scores = [x['highestProb'] for x in output]
        # threshold_predictions = [x['label'] if x['highestProb'] > 0.6 else 'undefined' for x in output]
        # print(classification_report(y_true=ground_truths, y_pred=threshold_predictions))
        # df = pandas.DataFrame({'intent': ground_truths, 'pred_intent': predictions, 'pred_score': scores, 'text': df_te.text})
        # df.to_json(os.path.join(test_model_path, 'test_2.json'), lines=True, orient='records')
        ######################### evaluating the prediction ends #########################


if __name__ == '__main__':
    main()
