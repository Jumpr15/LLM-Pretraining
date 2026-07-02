from transformers import PreTrainedModel

from .configuration_lightningtransformer import LightningTransformerModelConfig
from .lightningtransformer import LightningTransformer

class LightningTransformerModel(PreTrainedModel):
  config_class = LightningTransformerModelConfig

  def __init__(self, config):
    super().__init__(config)
    self.model = LightningTransformer(**config.cfg)
    self.post_init()
    