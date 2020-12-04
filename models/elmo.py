import allennlp.modules.elmo


class Elmo(allennlp.modules.elmo.Elmo):
    r"""
    A wrapper around AllenNLP's ELMo. Instead of outputting two types embeddings,
    we output just one, by summing up the two types of embeddings.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, *args, **kwargs):
        encoded_layers = super().forward(*args, **kwargs)
        return encoded_layers['elmo_representations'][0] + encoded_layers['elmo_representations'][1]
