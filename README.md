## LLM Pretraining Code 
Pretraining code for Jumpr/Nuwa-v2-10k-steps-ForCausalLM


### Model Architecture
- Used Fused Ops with Liger Kernel
- MHA with RoPE 
- RMS Pre-Norm 
- SwiGLU FFN with 3x expansion factor 
- Untied Embeddings
- Tokenizer is SmolLM-1.7B (ckpt: HuggingFaceTB/SmolLM-1.7B)

| | |
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

- WSD Scheduler (2k Warmup steps w/ Linearly increasing LR, Constant Max LR, Cosine Decay LR for final 0.1*Total steps)
- AdamW w/ default hyperparameters 

### Dataset 
- Used Dataset with streaming => Sequential Dataset Sampling 
- UltraFineWeb during warmup and stable phase
- UltraFineWeb-L3-Multi-Style-Synthetic during decay phase

### Training Stack
- Pytorch 
- Pytorch Lightning
- Liger Kernel
- WandB
- HF Datasets and Tokenizer and Transformers

### Training
- GPU: RTX A6000 Pro Ampere (~80hrs)