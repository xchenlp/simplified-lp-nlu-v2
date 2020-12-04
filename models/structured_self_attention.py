import torch


class StructuredSelfAttention(torch.nn.Module):
    r"""
    From https://github.com/kaushalshetty/Structured-Self-Attention/tree/master/attention, modified by Jian Wang.
    Applies self-attention sentence classifier on a list of word embeddings,
    to make predictions on sentence classification tasks

    Args:
        hidden_size: size of the input embeddings
        cls_num: number of classes in the output
        d_a: hidden size after the first linear layer
        r: number of the attention heads in the self attention mechanism

    Shape:
        embeddings: :math:`(batch_size, max_sent_len, hidden_size)`
        Output: :math:`(batch_size, cls_num)`
    """

    def __init__(self, hidden_size, cls_num, d_a=350, r=30):
        super(StructuredSelfAttention, self).__init__()
        self.lstm = torch.nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.linear_first = torch.nn.Linear(hidden_size, d_a, bias=False)
        self.linear_second = torch.nn.Linear(d_a, r, bias=False)
        self.cls_num = cls_num
        self.linear_final = torch.nn.Linear(hidden_size, self.cls_num)
        self.hidden_size = hidden_size
        self.r = r

    def forward(self, x):
        outputs, _ = self.lstm(x)
        x = torch.tanh(self.linear_first(outputs))
        x = self.linear_second(x)
        x = torch.nn.functional.softmax(x, dim=1)
        attention = x.transpose(1, 2)

        # @ for mat multiplication (see https://github.com/pytorch/pytorch/issues/1)
        sentence_embeddings = attention @ outputs
        avg_sentence_embeddings = torch.sum(sentence_embeddings, 1) / self.r
        return self.linear_final(avg_sentence_embeddings)
