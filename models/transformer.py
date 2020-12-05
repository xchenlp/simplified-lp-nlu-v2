import torch
from file_utils import transformer_base_model_names, transformer_encoders


class Transformer(torch.nn.Module):
    r"""
    A wrapper around Transformer. Instead of outputting two types embeddings,
    we output just one, by summing up the two types of embeddings.
    """
    def __init__(self, encoder_type: str, model_dir: str=None):
        super().__init__()
        self.encoder_type = encoder_type
        self.base_model_name = transformer_base_model_names[self.encoder_type]

        self.encoder = transformer_encoders[self.encoder_type]\
            .from_pretrained(model_dir if model_dir else self.base_model_name)
        ## ToDo: potentially load a fine-tuned encoder from state dict load_state_dict
        self.hidden_size = self.encoder.config.hidden_size
        for p in self.encoder.parameters():
            p.requires_grad = False

    def forward(self, message_attention_tuple):
        message_tensor, attention_mask = message_attention_tuple
        return self.encoder(message_tensor, attention_mask=attention_mask)[0]
