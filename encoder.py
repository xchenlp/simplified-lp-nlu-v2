import os
from models.elmo import Elmo
from models.fasttext import Fasttext
import logging
import torch
from models.transformer import Transformer
from file_utils import transformer_types
from torch import nn
from typing import Union, List
from models.elmo_tokenizer import ElmoTokenizer
from models.fasttext_tokenizer import FasttextTokenizer
from models.transformer_tokenizer import TransformerTokenizer


logger = logging.getLogger(__name__)


class Encoder(nn.Module):

## TODO: 
# 1) TransformerTokenizer will break if pad_to_max_sent_len is set to False.  
# 2) how to deal with empty str in forward
# 3) change to device to be a parent 

    def __init__(self, encoder_type: str, model_dir: str=None, 
        max_sent_len: int=100, pad_to_max_sent_len: bool=True,
        gpu_id: int=-1):
        """
        Args:
            encoder_type: encoder type to be used
            model_dir: (optional, defaults to None) load model from dir
            max_sent_len: (optional, defaults to 100) max length of tokens to encode
            pad_to_max_sent_len: (optional, defaults to True) pad each sentence to the max sentence length in the batch
            train: (optional) train or eval mode
        """
        super().__init__()
        self.tokenizer = self._get_tokenizer(encoder_type, max_sent_len, pad_to_max_sent_len)
        self.encoder = self._get_encoder(encoder_type, model_dir)
        device_str = f"cuda:{gpu_id}" if torch.cuda.is_available() and gpu_id != -1 else "cpu"
        logger.info(f'Device: {device_str}')
        self.device = torch.device(device_str)
        self.encoder.to(self.device)
        

    def forward(self, messages: Union[List[str], str]) -> torch.Tensor:
        if isinstance(messages, str):
            messages = [messages]
        message_tensor = self.tokenizer(messages)
        if isinstance(message_tensor, tuple): # for transformer-type encoder
            message_tensor = [item.to(self.device) for item in message_tensor]
        elif isinstance(message_tensor, torch.Tensor):
            # for elmo
            message_tensor = message_tensor.to(self.device)
        #else: for fasttext List[str]
        return self.encoder(message_tensor).to(self.device)
        

    def _get_tokenizer(self, encoder_type: str, max_sent_len: int, pad_to_max_sent_len: bool=False):
        r"""
        Get a Pytorch tokenizer for the corresponding encoder using the args from the SentenceClassifier class
        :param args: the SentenceClassifier class arguments
        :param model_dir: the path to the model being loaded
        :return: a Pytorch tokenizer model to tokenize str sentences
        """

        if encoder_type == 'elmo':
            return ElmoTokenizer(pad_to_max_sent_len, max_sent_len)
        elif encoder_type == 'fasttext':
            return FasttextTokenizer(pad_to_max_sent_len, max_sent_len)
        elif encoder_type in transformer_types:
            return TransformerTokenizer(encoder_type, max_sent_len, pad_to_max_sent_len)
        else:
            raise KeyError(f'''unknown encoder {encoder_type}''')


    def _get_encoder(self, encoder_type: str, model_dir: str):
        r"""
        Get a Pytorch encoder using the args from the SentenceClassifier class
        :param args: the SentenceClassifier class arguments
        :param model_dir: the path to the model being loaded
        :return: a Pytorch encoder model to generate word embeddings
        """
        if model_dir: 
            if encoder_type == 'elmo':
                options_file = os.path.join(model_dir, 'config.json')
                weights_file = os.path.join(model_dir, 'elmo.hdf5')
                model = Elmo(options_file, weights_file, 2, requires_grad=False, dropout=0.5)
                self._load_elmo_weights(model, model_dir)
            elif encoder_type == 'fasttext':
                options_file = os.path.join(model_dir, 'config.json')
                weights_file = os.path.join(model_dir, 'fasttext.pt')
                model = Fasttext(options_file, pickle_model_path=weights_file)
            elif encoder_type in transformer_types:
                model = Transformer(encoder_type=encoder_type, model_dir=os.path.join(model_dir, encoder_type))
            else:
                raise KeyError(f'''unknown encoder {encoder_type}''')
        else:
            if encoder_type in transformer_types:
                model = Transformer(encoder_type=encoder_type)
            else:
                raise Exception(f"need to provide a model dir for {encoder_type}")

        return model


    def _load_elmo_weights(self, model_dir):
        r"""
        Load the weights of a encoder model
        :param model: the encoder model that needs to load weights
        :param args: the SentenceClassifier class arguments
        :param model_dir: the path to the model being loaded
        """
        scalar_mix_0_weights_file = os.path.join(model_dir, 'encoder.scalar_mix_0.pt')
        scalar_mix_1_weights_file = os.path.join(model_dir, 'encoder.scalar_mix_1.pt')
        if os.path.exists(scalar_mix_0_weights_file) and os.path.exists(scalar_mix_1_weights_file):
            logger.info(f'''Loading ELMo embedding mixing layer 0 weights from {scalar_mix_0_weights_file}''')
            self.encoder.scalar_mix_0.load_state_dict(torch.load(scalar_mix_0_weights_file, map_location='cpu'))
            logger.info(f'''Loading ELMo embedding mixing layer 1 weights from {scalar_mix_1_weights_file}''')
            self.encoder.scalar_mix_1.load_state_dict(torch.load(scalar_mix_1_weights_file, map_location='cpu'))
        else:
            logger.info("Using pretrained ELMo")
