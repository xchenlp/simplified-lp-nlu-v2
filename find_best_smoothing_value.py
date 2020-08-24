from model import Model
import os
import pandas
from sklearn.metrics import classification_report, f1_score, accuracy_score
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = '-1'  # disable gpu. This is because on ca-gpu2, the cuDNN version is wrong for tensorflow


def main(args):
    f1s, accs = [], []
    es = [0.03, 0.04, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    for e in es:
        model = Model(word2vec_pkl_path='/data/cb_nlu_v2/vectors/wiki-news-300d-1M.pkl', config_path='config.yml', label_smoothing=e)
        test_model_path = f'/data/cb_nlu_{args.vertical}_cross_entropy_loss_label_smoothing/e_{e}'
        if not os.path.exists(test_model_path):
            os.makedirs(test_model_path)
        tr_set_path = f'/data/starter_pack_datasets/{args.vertical}/tr.json'
        te_set_path = f'/data/starter_pack_datasets/{args.vertical}/te.json'
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
        df_te.to_json(f'/data/cb_nlu_results/{args.vertical}/te_preds_xentroy_smoothing_{e}.json', orient='records', lines=True)
        print(classification_report(y_true=ground_truths, y_pred=threshold_predictions))
        f1s.append(f1_score(y_true=ground_truths, y_pred=threshold_predictions, average='macro'))
        accs.append(accuracy_score(y_true=ground_truths, y_pred=threshold_predictions))
        ######################### evaluating the prediction ends #########################
    max_f1 = max(f1s)
    max_acc = max(accs)
    max_f1_e = f1s.index(max_f1)
    max_acc_e = accs.index(max_acc)
    print(f'best macro f1 score {max_f1} is obtained at e={es[max_f1_e]}')
    print(f'best accuracy score {max_acc} is obtained at e={es[max_acc_e]}')

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vertical', required=True, type=str)
    args = parser.parse_args()
    main(args)
