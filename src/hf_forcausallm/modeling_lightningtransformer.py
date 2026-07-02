from transformers import PreTrainedModel

from .configuration_lightningtransformer import LightningTransformerModelConfig
from .lightningtransformer import LightningTransformer

class LightningTransformerModelForCausalLM(PreTrainedModel):
  config_class = LightningTransformerModelConfig

  def __init__(self, config):
    super().__init__(config)
    self.model = LightningTransformer(**config.cfg)
    self.use_cache = False
    self.post_init()
    
  def forward(self, input_ids, **kwargs):
    return self.model.forward(input_ids)