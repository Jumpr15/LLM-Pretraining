from transformers import PretrainedConfig

class LightningTransformerModelConfig(PretrainedConfig):
  model_type = "lightning_transformer"

  def __init__(self, cfg=None, **kwargs):
    self.cfg = cfg

    super().__init__(**kwargs)
