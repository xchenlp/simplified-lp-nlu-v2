import torch
from allennlp.modules.elmo import batch_to_ids
from typing import List


class ElmoTokenizer(torch.nn.Module):
    r"""
    Tokenizer for ELMo. Input is a list of str format sentence. Output is the tokenized text that fits the ELMo
    encoder's input format. Applies a cutoff to max_sent_len. One can choose whether to pad to max sentence length if
    the maximum sentence in the batch is less than max_sent_len
    """
    def __init__(self, pad_to_max_sent_len=False, max_sent_len=100):
        super().__init__()
        self.pad_to_max_sent_len = pad_to_max_sent_len
        self.max_sent_len = max_sent_len

    def forward(self, list_of_messages: List[str]) -> torch.Tensor:
        # TODO: adding a unittest testing padding to max_sent_len logic
        message_list = [message.split()[:self.max_sent_len] for message in list_of_messages]
        ## no padding for the sentences? 
        # is pad_to_max_sent_len padding for each sent or pad to batch size?
        if self.pad_to_max_sent_len:
            message_list.append(['n'] * self.max_sent_len)  # padding the message_list with a dummy message
            # then remove the padded sent character ids from the output? why? because of some weird behavior
            # regarding the char CNN? 
            message_tensor = batch_to_ids(message_list)[:-1, :, :]
        else:
            message_tensor = batch_to_ids(message_list)

        return message_tensor
