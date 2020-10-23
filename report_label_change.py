from model import Model
import os
import pandas
from sklearn.metrics import classification_report, f1_score, accuracy_score


os.environ["CUDA_VISIBLE_DEVICES"] = '-1'  # disable gpu. This is because on ca-gpu2, the cuDNN version is wrong for tensorflow


def main():
    # make two runs with the same dataset and compare label inconsistency 
    # and performance difference
    model = Model(word2vec_pkl_path='/data/cb_nlu_v2/vectors/wiki-news-300d-1M.pkl', config_path='config.yml')

    test_model_path = '/data/cb_nlu_test_model'
    tr_set_path = '/data/starter_pack_datasets/airlines/tr_80_per_class.json'
    te_set_path = '/data/starter_pack_datasets/airlines/te.json'

    labels_after_thresholding, labels_before_thresholding = [], []
    f1s_compare = []
    acc_compare = []
    num_runs = 2
    for i in range(num_runs):
        ######################### training #########################
        model.train(tr_set_path, test_model_path, stratified_split=True)
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
        labels_before_thresholding.append(predictions) 
        scores = [x['highestProb'] for x in output]
        threshold_predictions = [x['label'] if x['highestProb'] > 0.6 else 'undefined' for x in output]
        labels_after_thresholding.append(threshold_predictions)
        print(classification_report(y_true=ground_truths, y_pred=threshold_predictions))
        f1s_compare.append(f1_score(y_true=ground_truths, y_pred=threshold_predictions, average='macro'))
        acc_compare.append(accuracy_score(y_true=ground_truths, y_pred=threshold_predictions))
        df = pandas.DataFrame({'intent': ground_truths, 'pred_intent': predictions, 'pred_score': scores, 'text': df_te.text})
        df.to_json(os.path.join(test_model_path, f'test_{i}_airlines_80_per_class.json'), lines=True, orient='records')
        ######################### evaluating the prediction ends #########################

    # report label change and performance change
    percentage_label_change_before_thresholding = get_percent_label_change(*labels_before_thresholding)
    percentage_label_change_after_thresholding = get_percent_label_change(*labels_after_thresholding)
    f1_delta = get_performance_diff(*f1s_compare)
    acc_delta = get_performance_diff(*acc_compare)
    print(f"percentage of label change before thresholding: {percentage_label_change_before_thresholding}")
    print(f"percentage of label change after thresholding: {percentage_label_change_after_thresholding}")
    print(f"f1 delta: {f1_delta}")
    print(f"acc delta: {acc_delta}")


def get_percent_label_change(labels_1, labels_2):
    return sum([l1 != l2 for l1, l2 in zip(labels_1, labels_2)]) / len(labels_1)


def get_performance_diff(score_1, score_2):
    return abs(score_1 - score_2)

if __name__ == '__main__':
    main()
