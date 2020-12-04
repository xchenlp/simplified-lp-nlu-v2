import torch
import torch.nn.functional as F


class CNNClassifier(torch.nn.Module):
    r"""
    from git repo https://github.com/galsang/CNN-sentence-classification-pytorch/blob/master/model.py
    Applies CNN sentence classifier on a list of word embeddings, to make predictions on sentence classification tasks

    Args:
        hidden_size: size of the input and hidden embeddings
        cls_num: number of classes in the output
        max_sent_len: maximum sentence number. This version of CNN uses a fixed length for sentences
        dropout: the probability of dropping out an output value in the CNN output used to prevent overfitting in training. Doesn't matter in evaluation

    Shape:
        embeddings: :math:`(batch_size, max_sent_len, hidden_size)`
        Output: :math:`(batch_size, cls_num)`
    """
    def __init__(self, hidden_size, cls_num, max_sent_len, dropout=0.5):
        super(CNNClassifier, self).__init__()
        self.MAX_SENT_LEN = max_sent_len
        self.dropout = dropout
        self.WORD_DIM = hidden_size
        self.CLASS_SIZE = cls_num
        self.FILTERS = [3, 4, 5]
        self.FILTER_NUM = [100, 100, 100]
        self.IN_CHANNEL = 1

        assert (len(self.FILTERS) == len(self.FILTER_NUM))

        for i in range(len(self.FILTERS)):
            conv = torch.nn.Conv1d(self.IN_CHANNEL, self.FILTER_NUM[i], self.WORD_DIM * self.FILTERS[i], stride=self.WORD_DIM)
            # the f string is so interesting! So is the setattr function
            setattr(self, f'conv_{i}', conv)

        self.dropout_layer = torch.nn.Dropout(p=dropout)
        self.fc = torch.nn.Linear(sum(self.FILTER_NUM), self.CLASS_SIZE)

    def get_conv(self, i):
        return getattr(self, f'conv_{i}')

    def forward(self, embeddings):
        # in eval the last batch may not be of batch size
        embeddings_reshaped = embeddings.view(-1, 1, self.WORD_DIM * self.MAX_SENT_LEN)
        conv_results = [F.max_pool1d(F.relu(self.get_conv(i)(embeddings_reshaped)), self.MAX_SENT_LEN -
                                     self.FILTERS[i] + 1).view(-1, self.FILTER_NUM[i]) for i in
                        range(len(self.FILTERS))]
        x = torch.cat(conv_results, 1)
        x = self.dropout_layer(x)
        x = self.fc(x)
        return x
