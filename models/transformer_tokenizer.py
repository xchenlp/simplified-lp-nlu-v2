import torch
import torch.nn
from typing import List, Tuple
import numpy

from lp_nlu.file_utils import transformer_base_model_names, transformer_tokenizers


class TransformerTokenizer(torch.nn.Module):
    r"""
    Tokenizer for Transformers. Input is a list of str format sentence. 
    Output is the tokenized text that fits the Transformer encoder's input format.
    Applies a cutoff to max_sent_len. One can choose whether to pad to max sentence length if
    the maximum sentence in the batch is less than max_sent_len
    """

    def __init__(self, encoder_type: str, max_sent_len: int=100, 
        pad_to_max_sent_len: bool=False):
        """
        Args:
            max_sent_len: the max length of the wordpiece tokens to encode in a sentence
        """
        super().__init__()
        self.max_sent_len = max_sent_len
        self.pad_to_max_sent_len = pad_to_max_sent_len
        self.encoder_type = encoder_type
        self.base_model_name = transformer_base_model_names[self.encoder_type]
        self.tokenizer = transformer_tokenizers[self.encoder_type].from_pretrained(self.base_model_name)


    def forward(self, list_of_messages: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        # output = self.tokenizer.batch_encode_plus(list_of_messages, truncation=True,
        #     max_length=self.max_sent_len, padding=self.pad_to_max_sent_len)
        # message_tensor = torch.tensor(output.get('input_ids'))
        # attention_mask = torch.tensor(output.get('attention_mask'))
        message_tensor = numpy.zeros((len(list_of_messages), self.max_sent_len), dtype=numpy.int64)
        attention_mask = numpy.zeros((len(list_of_messages), self.max_sent_len), dtype=numpy.int64)
        for i, message in enumerate(list_of_messages):
            message = self.tokenizer.encode(message, add_special_tokens=True, max_length=self.max_sent_len)
            msg_len = len(message)
            message_tensor[i, :msg_len] = message[:msg_len]
            attention_mask[i, :msg_len] = 1
        message_tensor = torch.tensor(message_tensor)
        attention_mask = torch.tensor(attention_mask)
        return message_tensor, attention_mask
        return message_tensor, attention_mask
