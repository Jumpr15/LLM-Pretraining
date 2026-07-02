from transformers import PretrainedConfig

class LightningTransformerModelConfig(PretrainedConfig):
  model_type = "lightning_transformer"

  def __init__(self, cfg=None, **kwargs):
    self.cfg = cfg
    self.num_hidden_layers = cfg["block_num"]

    super().__init__(**kwargs)
