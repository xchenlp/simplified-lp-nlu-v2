from model import Model
import os
import pandas
from sklearn.metrics import classification_report


#os.environ["CUDA_VISIBLE_DEVICES"] = '-1'  # disable gpu. This is because on ca-gpu2, the cuDNN version is wrong for tensorflow


def main():
    model = Model(word2vec_pkl_path='/home/xchen/models/wiki-news-300d-1M.pkl', config_path='config.yml', encoder_type='transformer', 
            label_smoothing=0, gpu_id=0)

    test_model_path = '/home/xchen/data/cb_nlu_test_model'
    tr_set_path = '/home/xchen/data/finserv_generic_tr.json'
    te_set_path = '/home/xchen/data/finserv_generic_te.json'

    ######################### training #########################
    print("start training")
    #set to train with early stopping on the average loss on a validation set obtained by a stratifed split of the training data with a fraction of 10% (default)
    model.train(tr_set_path, test_model_path, stratified_split=True, early_stopping=True)
     
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
    scores = [x['highestProb'] for x in output]
    threshold_predictions = [x['label'] if x['highestProb'] > 0.6 else 'undefined' for x in output]
    print(classification_report(y_true=ground_truths, y_pred=threshold_predictions))
    #df = pandas.DataFrame({'intent': ground_truths, 'pred_intent': predictions, 'pred_score': scores, 'text': df_te.text})
    #df.to_json(os.path.join(test_model_path, 'test_4_airlines_80_per_class.json'), lines=True, orient='records')
    ######################### evaluating the prediction ends #########################
    

if __name__ == '__main__':
    main()

