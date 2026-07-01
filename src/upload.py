import torch
import click
import yaml
from safetensors.torch import save_model, load_model

from hf_wrapper.configuration_lightningtransformer import LightningTransformerModelConfig
from hf_wrapper.modeling_lightningtransformer import LightningTransformerModel

@click.command()
@click.argument("model_ckpt_path")
@click.argument("hf_model_name")
@click.argument('upload_config_file')
def main(model_ckpt_path, hf_model_name, upload_config_file):
     with open(upload_config_file, 'r') as f:
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
     
     checkpoint = torch.load(model_ckpt_path, weights_only=False)
     # if checkpoint['hyper_parameters']['tie_weights'] and torch.allclose(checkpoint['state_dict']['embed_proj.weight'], checkpoint['state_dict']['token_embed.weight']): # If tie_weights is enabled in model and token_embed + embed_proj are equal
     #      print(list(checkpoint['state_dict']))
     #      # print(checkpoint['state_dict']['token_embed'])
     #      # print(checkpoint['state_dict']['embed_proj.weight'])
     #      # checkpoint['state_dict']['embed_proj.weight'].copy_(checkpoint['state_dict']['token_embed.weight'])
     #      # checkpoint['state_dict']['token_embed.weight'].copy_(checkpoint['state_dict']['embed_proj.weight'])

     
     # # this is only for save_model not safe_file
     model.model.load_state_dict(checkpoint['state_dict']) # Lightning ckpts save Weights as OrderedDicts => Convert into state_dict for safetensors loading compatibility
     
     save_model(model.model, "model.safetensors") # Save to dir as safetensors format
     
     load_model(model.model, "model.safetensors") # Load new safetensors file over existing torch state_dict
     
     model.push_to_hub(hf_model_name)

# needs path to ckpt and hf model name args
# example usage uv run upload.py model_ckpts/epoch=0-step=5.ckpt HF_compatibility_test ci_config.yaml
if __name__ == '__main__':
     main()