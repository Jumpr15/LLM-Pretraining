import torch
import click
import yaml
from safetensors.torch import save_model, load_model

from hf_automodel.configuration_lightningtransformer import LightningTransformerModelConfig
from hf_automodel.modeling_lightningtransformer import LightningTransformerModel

from hf_forcausallm.modeling_lightningtransformer import LightningTransformerModelForCausalLM

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
     }
     
     LightningTransformerModelConfig.register_for_auto_class()
     LightningTransformerModel.register_for_auto_class("AutoModel")
     LightningTransformerModelForCausalLM.register_for_auto_class("AutoModelForCausalLM")
     
     config = LightningTransformerModelConfig(model_config)
     automodel = LightningTransformerModel(config)
     causalmodel = LightningTransformerModelForCausalLM(config)
     
     config.auto_map = {
          "AutoConfig": "configuration_lightningtransformer.LightningTransformerModelConfig",
          "AutoModel": "modeling_lightningtransformer.LightningTransformerModel",
          "AutoModelForCausalLM": "modeling_lightningtransformer.LightningTransformerModelForCausalLM"
     }
     
     checkpoint = torch.load(model_ckpt_path, weights_only=False)

     # this is only for save_model not safe_file
     automodel.model.load_state_dict(checkpoint['state_dict']) # Lightning ckpts save Weights as OrderedDicts => Convert into state_dict for safetensors loading compatibility
     causalmodel.model.load_state_dict(checkpoint['state_dict'])
     
     save_model(automodel.model, "automodel.safetensors") # Save to dir as safetensors format
     save_model(causalmodel.model, "causalmodel.safetensors")
     
     load_model(automodel.model, "automodel.safetensors") # Load new safetensors file over existing torch state_dict
     load_model(causalmodel.model, "causalmodel.safetensors")
     
     automodel.push_to_hub(f'{hf_model_name}-AutoModel')
     causalmodel.push_to_hub(f'{hf_model_name}-ForCausalLM')

# needs path to ckpt and hf model name args
# example usage uv run upload.py model_ckpts/epoch=0-step=5.ckpt HF_compatibility_testv3 configs/ci_config.yaml
if __name__ == '__main__':
     main()