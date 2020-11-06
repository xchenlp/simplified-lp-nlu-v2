from model import Model
import os
import pandas as pd
from sklearn.metrics import classification_report, f1_score, accuracy_score
import numpy as np
import itertools

os.environ["CUDA_VISIBLE_DEVICES"] = '-1'  # disable gpu. This is because on ca-gpu2, the cuDNN version is wrong for tensorflow


def main():
    label_variance_before, label_variance_after, F1s, ACCs, f1_stds, acc_stds, var_std_before, var_std_after \
     = [], [], [], [], [], [], [], [] 
    e_values = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    test_model_path = '/data/cb_nlu_test_model_with_early_stopping'
    tr_set_path = '/data/deep-sentence-classifiers/preprocessed_data/Telstra/tr_5_intents_40_messages_per_intent.json'
    te_set_path = '/data/deep-sentence-classifiers/preprocessed_data/Telstra/te.json'
    print(f'training data: {tr_set_path}')
    for e in e_values:
        print(f'e={e}')
        model = Model(word2vec_pkl_path='/data/cb_nlu_v2/vectors/wiki-news-300d-1M.pkl', config_path='config.yml', label_smoothing=e)

        labels_after_thresholding, labels_before_thresholding = [], []
        f1s_compare = []
        acc_compare = []
        num_runs = 2
        for i in range(num_runs):
            ######################### training #########################
            model.train(tr_set_path, test_model_path, stratified_split=True, early_stopping=True)
            ######################### training ends #########################

            ######################### loading #########################
            #model.load(test_model_path)
            ######################### loading ends #########################

            ######################### predicting #########################
            df_te = pd.read_json(te_set_path, lines=True)
            output = model.predict(list(df_te.text))
            ######################### predicting ends #########################

            ######################### evaluating the prediction #########################
            ground_truths = list(df_te.intent)
            predictions = [x['label'] for x in output]
            labels_before_thresholding.append(predictions) 
            # scores = [x['highestProb'] for x in output]
            threshold_predictions = [x['label'] if x['highestProb'] > 0.6 else 'undefined' for x in output]
            labels_after_thresholding.append(threshold_predictions)
            # print(classification_report(y_true=ground_truths, y_pred=threshold_predictions))
            f1s_compare.append(f1_score(y_true=ground_truths, y_pred=threshold_predictions, average='macro'))
            acc_compare.append(accuracy_score(y_true=ground_truths, y_pred=threshold_predictions))
            #df = pd.DataFrame({'intent': ground_truths, 'pred_intent': predictions, 'pred_score': scores, 'text': df_te.text})
            #df.to_json(os.path.join(test_model_path, f'test_{i}_airlines_80_per_class.json'), lines=True, orient='records')
            ######################### evaluating the prediction ends #########################

        # report label change and performance change
        percentage_label_change_before_thresholding, std_before = get_percent_label_change(labels_before_thresholding)
        percentage_label_change_after_thresholding, std_after = get_percent_label_change(labels_after_thresholding)
        f1_stds.append(np.std(f1s_compare))
        acc_stds.append(np.std(acc_compare))
        F1s.append(np.mean(f1s_compare))
        ACCs.append(np.mean(acc_compare))
        label_variance_before.append(percentage_label_change_before_thresholding)
        label_variance_after.append(percentage_label_change_after_thresholding)
        var_std_before.append(std_before)
        var_std_after.append(std_after)

        print(f"percentage of label change before thresholding: {percentage_label_change_before_thresholding}")
        print(f"percentage of label change after thresholding: {percentage_label_change_after_thresholding}")
        print(f"mean F1: {np.mean(f1s_compare)}")
        print(f'mean acc: {np.mean(acc_compare)}')

    # store the dataframe
    df_stats = pd.DataFrame({
        "e_value": e_values,
        "mean_macro_f1": F1s,
        "mean_acc": ACCs,
        "label_change_before_thresh": label_variance_before,
        "label_change_after_thresh": label_variance_after,
        "macro_f1_std": f1_stds,
        "acc_std": acc_stds,
        "label_change_before_thresh_std": var_std_before,
        "label_change_after_thresh_std": var_std_after
        })
    training = tr_set_path.split('/')[-1][:-5]
    df_stats.to_json(f'/data/lp-nlu-early-stopping-LS/{training}_stats.json', lines=True, orient='records')

        # print(f"f1 delta: {f1_delta}")
        # print(f"acc delta: {acc_delta}")

def get_percent_label_change(labels):
    deltas = []
    for labels_1, labels_2 in itertools.combinations(labels, 2):
        delta = sum([l1 != l2 for l1, l2 in zip(labels_1, labels_2)]) / len(labels_1)
        deltas.append(delta)
    return (np.mean(deltas), np.std(deltas))

if __name__ == '__main__':
    main()
