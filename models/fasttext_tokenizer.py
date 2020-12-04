import torch.nn
from typing import List


class FasttextTokenizer(torch.nn.Module):
    r"""
    Tokenizer for FastText encoder. Input is a list of str format sentence. Output is the tokenized text that fits the FastText
    encoder's input format. Applies a cutoff to max_sent_len. One can choose whether to pad to max sentence length if
    the maximum sentence in the batch is less than max_sent_len
    """
    def __init__(self, pad_to_max_sent_len=False, max_sent_len=100):
        super().__init__()
        self.pad_to_max_sent_len = pad_to_max_sent_len
        self.max_sent_len = max_sent_len

    def forward(self, list_of_messages) -> List[List[str]]:
        # how to pad?
        list_of_tokenized_messages = []
        for i, text in enumerate(list_of_messages):
            tokenized_message = text.split()[:self.max_sent_len]
            list_of_tokenized_messages.append(tokenized_message)
        if self.pad_to_max_sent_len:
            # padding the first message to self.max_sent_len is enough
            list_of_tokenized_messages[0] = list_of_tokenized_messages[0] + ['|||PAD|||'] * (self.max_sent_len - len(list_of_tokenized_messages[0]))
        return list_of_tokenized_messages
