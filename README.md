## LLM Pretraining Code 
Pretraining code for Jumpr/Nuwa-v2-10k-steps-ForCausalLM

### Usage 
- Requires CUDA/ROCm compatible device (liger kernel dep)
- Uses Tmux for detachable and persistent sessions (model_train session created in setup.sh)
- Requires HF and WandB accounts w/ API Keys

Install dependencies and creates detachable tmux session to run training in
```bash
bash setup.sh
```

Runs training for a model based on preconfigured parameters in referenced yaml file
```bash
uv run train.py pretrain_config.yaml
```

Uploads model weights as safetensors to HF model (Have HF_TOKEN env var exported first)
```bash
uv run upload.py <path_to_checkpoint> <hf_model_name> pretrain_config.yaml
```

### Issues
- HF_Rsync to bucket callback causes processor memory problems, so must manually upload ckpts
- No fallback for non-Cuda/Rocm devices
- No stateful dataloader (If using sequential dataset processing, restarting training even on .ckpts causes dataset to restart from beginning)
- Better to Use UFW-10B Sample Dataset how dataset shuffling (CerebellumKing/Ultra-FineWeb-10B)

### Model Architecture
- Used Fused Ops with Liger Kernel
- MHA with RoPE 
- RMS Pre-Norm 
- SwiGLU FFN with 3x expansion factor 
- Untied Embeddings
- Tokenizer is SmolLM-1.7B (ckpt: HuggingFaceTB/SmolLM-1.7B)

|---|---|
| Parameters | ~483M |
| Layers | 28 |
| Hidden size | 1024 |
| Attention heads | 16 (head dim 64) |
| Context length | 1024 |
| Vocab size | 49,152 |
| Grad Clip Val | 1.0 |
| Iterations | 280k / 70k True steps (After accounting for batch acc)|
| Max LR | 2.5e-4 |
| Precision | Full bf-16 |
| Batch Size | 32 in warmup and stable => 36 in decay |
| Batch Acc | 4 |
| Optimizer | AdamW w/ default hyperparameters  |
| LR Scheduler | WSD Scheduler (2k Warmup steps w/ Linearly increasing LR, Constant Max LR, Cosine Decay LR for final 10% of Total steps) |

### Dataset 
- Used Dataset with streaming => Sequential Dataset Sampling 
- UltraFineWeb during warmup and stable phase (openbmb/Ultra-FineWeb split: en)
- UltraFineWeb-L3-Multi-Style-Synthetic during decay phase (openbmb/Ultra-FineWeb-L3 subset: Ultra-Fine-Web-L3-en-Multi-Style-Synthetic)

### Training Stack
- Pytorch 
- Pytorch Lightning
- Liger Kernel
- WandB
- HF Datasets and Tokenizer and Transformers

### Training
- GPU: RTX A6000 Pro Ampere (~80hrs)