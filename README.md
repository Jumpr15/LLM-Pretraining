## LLM Pretraining Code 

### Model Architecture
- Uses MHA with RoPE 
- RMS Pre-Norm 
- SwiGLU FFN with 3x expansion factor 
- Untied Embeddings
- Tokenizer is SmolLM-1.7B (ckpt: HuggingFaceTB/SmolLM-1.7B)

### Hyperparameters

- Max LR of 2.5e-4
- WSD Scheduler (2k Warmup steps w/ Linearly increasing LR, Constant Max LR, Cosine Decay LR for final 0.1*Total steps)
- AdamW w/ default hyperparameters 

- Trained with full bf-16 precision
- bsz_size 32 in warmup and stable phase => bsz_size increased to 36 in decay
- batch_acc 4 throughout training
