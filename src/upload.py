import torch
import click
import yaml
from safetensors.torch import save_file, load_file

from hf_wrapper.configuration_lightningtransformer import LightningTransformerModelConfig
from hf_wrapper.modeling_lightningtransformer import LightningTransformerModel

@click.command()
@click.argument("model_ckpt_path")
@click.argument("hf_model_name")
def main(model_ckpt_path, hf_model_name):
     with open('train_config.yaml', 'r') as f:
          config = yaml.safe_load(f)  
          
          batch_size = int(config['batch_size'])
          seq_len = int(config['seq_len'])
          embed_dims = int(config['embed_dims'])
          head_size = int(config['head_size'])
          num_heads = int(config['num_heads'])
          block_num = int(config['block_num'])
          vocab_size = int(config['vocab_size'])
          lr = float(config['lr'])
          iterations = int(config['iterations'])
          enable_liger_kernel = bool(config['enable_liger_kernel'])
          tie_weights = bool(config['tie_weights'])
     
     model_config = {
          "batch_size": batch_size,
          "seq_len": seq_len,
          "embed_dims": embed_dims,
          "head_size": head_size,
          "num_heads": num_heads,
          "block_num": block_num,
          "vocab_size": vocab_size,
          "lr": lr,
          "iterations": iterations,
          "use_liger": enable_liger_kernel,
          "tie_weights": tie_weights
     }
     
     LightningTransformerModelConfig.register_for_auto_class()
     LightningTransformerModel.register_for_auto_class("AutoModel")
     
     config = LightningTransformerModelConfig(model_config)
     model = LightningTransformerModel(config)
     
     checkpoint = torch.load(model_ckpt_path, weights_only=True)
     state_dict = checkpoint['state_dict']
     save_file(state_dict, f"model-{checkpoint}.safetensors")
     
     model.model.load_state_dict(load_file(f"model-{checkpoint}.safetensors"))
     
     model.push_to_hub(hf_model_name)

# needs path to ckpt and hf model name args
if __name__ == '__main__':
     main()