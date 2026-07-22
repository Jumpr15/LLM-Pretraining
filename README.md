## LLM Pretraining Code 
Pretraining code for Jumpr/Nuwa-v2-10k-steps-ForCausalLM


### Model Architecture
- Used Fused Ops with Liger Kernel
- MHA with RoPE 
- RMS Pre-Norm 
- SwiGLU FFN with 3x expansion factor 
- Untied Embeddings
- Tokenizer is SmolLM-1.7B (ckpt: HuggingFaceTB/SmolLM-1.7B)

### Dataset 
- Used Dataset with streaming => Sequential Dataset Sampling 
- UltraFineWeb during warmup and stable phase
- UltraFineWeb-L3-Multi-Style-Synthetic during decay phase

### Hyperparameters
- Max LR of 2.5e-4
- WSD Scheduler (2k Warmup steps w/ Linearly increasing LR, Constant Max LR, Cosine Decay LR for final 0.1*Total steps)
- AdamW w/ default hyperparameters 

- Trained with full bf-16 precision
- bsz_size 32 in warmup and stable phase => bsz_size increased to 36 in decay
- batch_acc 4 throughout training

### Training Stack
- Pytorch 
- Pytorch Lightning
- Liger Kernel
- WandB
- HF Datasets and Tokenizer and Transformers

### Training
- GPU: RTX A6000 Pro Ampere (~80hrs)