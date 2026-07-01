from transformers import PreTrainedModel

from .configuration_lightningtransformer import LightningTransformerModelConfig
from .lightningtransformer import LightningTransformer

class LightningTransformerModel(PreTrainedModel):
  config_class = LightningTransformerModelConfig
  _tied_weights_keys = {
    "model.embed_proj.weight": "model.token_embed.weight"
  }

  def __init__(self, config):
    super().__init__(config)
    self.model = LightningTransformer(**config.cfg)
    self.post_init()
    
  # # hooks for input/output embedding layers => required for interpreting tied embeddings
  # def get_input_embeddings(self):
  #     return self.model.token_embed

  # def set_input_embeddings(self, value):
  #     self.model.token_embed = value

  # def get_output_embeddings(self):
  #     return self.model.embed_proj

  # def set_output_embeddings(self, value):
  #     self.model.embed_proj = value

  def forward(self, input_ids, **kwargs):
    return self.model.forward(input_ids)