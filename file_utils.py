from transformers import BertTokenizer, RobertaTokenizer, DistilBertTokenizer, AlbertTokenizer
from transformers import DistilBertModel, BertModel, RobertaModel, AlbertModel

transformer_types = {'bert', 'roberta', 'distilbert', 'albert', 'distilmbert'}

transformer_base_model_names = {'bert': 'bert-base-uncased',
                                'roberta': 'roberta-base',
                                'distilbert': 'distilbert-base-uncased',
                                'distilmbert': 'distilbert-base-multilingual-cased',
                                'albert': 'albert-base-v2'}

transformer_tokenizers = {'bert': BertTokenizer,
                          'roberta': RobertaTokenizer,
                          'distilbert': DistilBertTokenizer,
                          'distilmbert': DistilBertTokenizer,
                          'albert': AlbertTokenizer}

transformer_encoders = {'bert': BertModel,
                        'roberta': RobertaModel,
                        'distilbert': DistilBertModel,
                        'distilmbert': DistilBertModel,
                        'albert': AlbertModel}
