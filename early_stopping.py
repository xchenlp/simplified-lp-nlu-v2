import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from tensorflow.keras.callbacks import Callback
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import numpy as np

class EarlyStoppingAtMaxMacroF1(Callback):
    """Stop training when the loss is at its min, i.e. the loss stops decreasing.

  Arguments:
      patience: Number of epochs to wait after min has been hit. After this
      number of no improvement, training stops.
  """

    def __init__(self, validation, patience=0):
        super().__init__()
        self.patience = patience
        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None
        self.validation = validation

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best = 0
        
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
        self.val_accuracies = []

    def on_epoch_end(self, epoch, logs=None):
        val_targ = self.validation[1]   
        val_predict = self.model.predict_classes(self.validation[0])        
        
        val_f1 = f1_score(val_targ, val_predict, average='macro')
        val_recall = recall_score(val_targ, val_predict, average='macro')         
        val_precision = precision_score(val_targ, val_predict, average='macro')
        val_accuracy = accuracy_score(val_targ, val_predict)

        self.val_f1s.append(round(val_f1, 6))
        self.val_recalls.append(round(val_recall, 6))
        self.val_precisions.append(round(val_precision, 6))
        self.val_accuracies.append(round(val_accuracy, 6))
        
        #ToDo: may just append to the logs, instead of writing new lists
        logs['val_f1'] = round(val_f1, 6)
        logs['val_recall'] = round(val_recall, 6)
        logs['val_precision'] = round(val_precision, 6)
        logs['val_acc'] = round(val_accuracy, 6)       

        print(f' — val_f1: {val_f1} — val_precision: {val_precision}, — val_recall: {val_recall}, - val_acc: {val_accuracy}')

        current = val_f1
        if np.greater(current, self.best):
            self.best = current
            self.wait = 0
            # Record the best weights if current results is better (less).
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print("Restoring model weights from the end of the best epoch.")
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))
