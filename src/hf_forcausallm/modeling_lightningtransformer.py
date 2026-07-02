from transformers import PreTrainedModel
from transformers.generation import GenerationMixin

from .configuration_lightningtransformer import LightningTransformerModelConfig
from .lightningtransformer import LightningTransformer


class LightningTransformerModelForCausalLM(PreTrainedModel, GenerationMixin):
  config_class = LightningTransformerModelConfig

  def __init__(self, config):
    super().__init__(config)
    self.model = LightningTransformer(**config.cfg)
    self.use_cache = False
    self.num_hidden_layers = config.cfg["block_num"]
    self.post_init()
    
  def forward(self, input_ids, **kwargs):
    return self.model.forward(input_ids)