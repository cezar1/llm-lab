# LLM Lab: Learn LLM Concepts the Hard (Real) Way

A hands-on curriculum for mastering core LLM concepts through implementation, visualization, and ablation studies. Each project teaches one concept by building it from scratch, visualizing how it works, and breaking it to understand it.

**Philosophy:** Code → Plot → Break → Learn. No theory without implementation.

---

## Curriculum Map

### 00. Utils
**Utilities for all projects**
- `plotting.py`: Visualization helpers (heatmaps, scatter plots)
- `data.py`: Data loading utilities

### 01. Tokenization & Embeddings
**Learn:** Byte-pair encoding, vocabulary building, token visualization
- `bpe.py`: Hand-coded BPE algorithm with vocabulary building
- `embeddings.py`: One-hot vs learned embeddings, cosine similarity analysis
- `demo.py`: Full demo with compression ratios, embedding ablations

**Key Insights:**
- BPE learns frequent subword patterns through iterative merging
- More merges → better compression with diminishing returns
- Learned embeddings capture similarity better than one-hot
- Token IDs can be visualized to understand model input

---

### 02. Positional Embeddings
**Learn:** Sinusoidal, Learned, RoPE, ALiBi encodings
- `positional_embeddings.py`: All 4 methods with visualization
- `demo.py`: Heatmaps, similarity matrices, extrapolation tests, ablations

**Key Insights:**
- Sinusoidal: Fixed, excellent extrapolation
- Learned PE: Flexible but poor extrapolation
- RoPE: Rotary encoding, excellent for long contexts
- ALiBi: No PE vectors, adds bias to attention instead
- RoPE is the modern standard for long-context models

---

### 03. Attention Basics
**Learn:** Single-token attention, multihead mechanisms, causality
- `attention.py`: Dot-product attention, multihead implementation, causal masking
- `demo.py`: Visualization of weight matrices, head patterns, causality tests

**Key Insights:**
- Attention = softmax(Q@K^T / sqrt(d)) @ V
- Multiple heads capture different relationships
- Different heads specialize (diversity principle)
- Causal masking prevents attending to future tokens
- Scaling (1/sqrt(d)) prevents attention saturation

---

### 04. Transformer Block
**Learn:** Stacking attention + FFN with LayerNorm and residuals
- `transformer.py`: Single block, n-block "mini-former", analysis tools
- `demo.py`: Information flow visualization, layer-wise analysis, component ablations

**Key Insights:**
- Residual connections: x → norm → attn → + x → norm → ffn → + x
- LayerNorm stabilizes training by normalizing activations
- Each block can be understood as: attention (What to attend to) + FFN (What to compute)
- Stacking blocks increases model capacity for complex reasoning

---

### 05. Sampling Parameters
**Learn:** Temperature, Top-K, Top-P decoding strategies
- `sampler.py`: Implementations with entropy and sparsity analysis
- `demo.py`: Interactive ablations, distribution comparisons, recommendations

**Key Insights:**
- Temperature=0 is argmax (deterministic)
- Higher temperature → higher entropy → more diversity
- Top-K: fix number of candidates
- Top-P (nucleus): fix cumulative probability mass
- Use either top-K or top-P, not both

**Guidelines:**
- Deterministic: temp=0.0
- Focused: temp=0.7, top_k=10, top_p=0.9
- Creative: temp=1.5-2.0, top_k=100, top_p=0.99

---

### 06. KV Cache
**Learn:** Fast inference through cached keys/values
- `cache.py`: Cache implementation, speedup benchmarking, memory analysis
- `demo.py`: Speedup visualization, memory tradeoffs, efficiency analysis

**Key Insights:**
- Without cache: O(seq_len²) attention at each step
- With cache: O(seq_len) computation per step
- Speedup grows with sequence length (10x @ len=100, 100x+ @ len=1000)
- Memory cost worth it for nearly all generation scenarios
- Advanced: sliding window cache, sparse cache, quantized cache

---

### 07. Long-Context Tricks
**Learn:** Handling long sequences efficiently
- `sliding_window.py`: Sliding window attention implementation
- Includes memory-efficient variants and context length ablations

**Key Insights:**
- Full attention is O(n²) — unfeasible for very long sequences
- Sliding window: attend only to recent k tokens
- Computes attention loss on increasingly long documents
- Context collapse: model performance drops beyond training length
- Extrapolation is hard; better to train on long sequences

---

### 08. Mixture of Experts (MoE)
**Learn:** Dynamic routing of tokens to expert networks
- `moe_layer.py`: Router implementation, expert utilization tracking
- FLOP savings analysis, load balancing strategies

**Key Insights:**
- Route tokens to different FFN experts based on input
- Sparse activation: only some experts fire per token
- FLOP savings with maintained model capacity
- Load balancing: avoid all tokens going to same expert
- Enables much larger models with same compute

---

### 09. Grouped Query Attention (GQA)
**Learn:** Share key/value heads across query heads
- `gqa.py`: Convert multihead to grouped-query attention
- Speed/memory comparisons, group count ablations

**Key Insights:**
- Standard MHA: separate K,V for each head
- GQA: groups of query heads share one K,V head
- Reduces KV cache size (crucial for long-context)
- Maintains most quality with ~4x smaller cache
- Modern inference: GQA is becoming standard

---

### 10. Normalization & Activations
**Learn:** LayerNorm, RMSNorm, GELU, SwiGLU, etc.
- `layers.py`: Implementations of all normalization and activation functions
- Ablation studies on impact to training stability and speed

**Key Insights:**
- LayerNorm: normalize to mean=0, std=1 per example
- RMSNorm: simpler, no centering, widely used in modern models
- GELU: smooth approximation of ReLU
- SwiGLU: Swish-gated FFN, better than standard MLPs
- Different combinations affect convergence speed and final performance

---

### 11. Pretraining Objectives
**Learn:** Masked LM, Causal LM, Prefix LM
- `objectives.py`: Different loss functions and training targets
- `demo.py`: Loss curve comparisons, masking strategies, convergence analysis

**Key Insights:**
- Causal LM: predict next token (GPT-style)
- Masked LM: predict masked tokens bidirectionally (BERT-style)
- Prefix LM: see prefix, predict suffix (hybrid approach)
- Causal: good for generation, MLM: good for encoding
- Masking rate (~15%) impacts learning efficiency

---

### 12. Finetuning & RLHF
**Learn:** Supervised finetuning, instruction tuning, reward modeling
- `finetuning.py`: Training curves for different approaches
- `demo.py`: Reward signal dynamics, format effectiveness analysis

**Key Insights:**
- Supervised FT: fast convergence, task-specific
- Instruction TG: better generalization to new tasks
- RLHF: learns from human preference signals
- Instruction format matters (QA > plain)
- RLHF trade-off: reward maximization vs staying close to base model (KL penalty)

---

### 13. Scaling Laws
**Learn:** Model size impact on loss, compute, memory
- `scaling.py`: Compute scaling curves, Chinchilla law analysis
- `demo.py`: Loss vs parameters, VRAM/time requirements, Pareto frontiers

**Key Insights:**
- Loss follows power law: loss ≈ a + b/N^0.07
- Chinchilla optimal: equal compute for model & data
- Bigger models: better sample efficiency, but more compute
- Optimal size depends on time/memory budget (Pareto frontier)
- Early stopping matters; don't train to convergence always

---

### 14. Quantization
**Learn:** Post-training and quantization-aware training
- `quantization.py`: PTQ, QAT implementations
- `demo.py`: Accuracy vs bit-width tradeoffs, GGUF/AWQ export

**Key Insights:**
- PTQ: quantize after training, fast but loses accuracy
- QAT: quantize during training, better quality
- 8-bit: minimal accuracy loss, 2x memory savings
- 4-bit: more aggressive, ~99% of performance
- Quantization crucial for on-device/edge inference

---

### 15. Inference Stacks
**Learn:** HuggingFace vs vLLM vs ExLlama
- `inference.py`: Performance profiling across frameworks
- `demo.py`: Throughput/latency comparison, recommendations

**Key Insights:**
- HF: Flexible, easy, baseline performance
- vLLM: 5-10x faster, best for production
- ExLlama: Fastest for Llama, quantized only
- Throughput scales with batch size
- For production: vLLM is typically optimal

---

### 16. Synthetic Data
**Learn:** Generate, augment, and use synthetic training data
- `generate.py`: Data generation, noise addition, deduplication
- `demo.py`: Learning curves on real vs synthetic data

**Key Insights:**
- Synthetic data can match real data quality if carefully designed
- Noise injection helps robustness
- Deduplication prevents overfitting
- Train/val/test splits crucial
- Synthetic data useful for pretraining and augmentation

---

## Quick Start

### Install Dependencies
```bash
pip install torch torch.nn matplotlib numpy scikit-learn
```

### Run Any Project
```bash
cd 01_tokenization
python demo.py

cd ../03_attention_basics
python demo.py

cd ../13_scaling_laws
python demo.py
```

### Project Structure
Each project has:
- `*.py`: Core implementations
- `demo.py`: Complete demo with visualizations
- Optional: Ablation studies and comparisons

---

## Learning Path

**Foundational (Start Here)**
1. 01_tokenization - Understand token representation
2. 02_positional_embeddings - Learn position encoding methods
3. 00_utils - Understand visualization tools

**Core Concepts**
4. 03_attention_basics - Master attention mechanism
5. 04_transformer_block - Stack blocks into models
6. 05_sampling - Generate text intelligently

**Advanced Architecture**
7. 10_normalization - Stabilize training
8. 08_moe - Scale with sparse computation
9. 09_gqa - Efficient attention variants

**Optimization & Deployment**
10. 06_kv_cache - Fast inference
11. 14_quantization - Compress models
12. 15_inference_stacks - Deploy at scale

**Training & Performance**
13. 11_pretraining_objectives - Different training targets
14. 12_finetuning_rlhf - Adapt models to tasks
15. 13_scaling_laws - Understand model capacity
16. 07_long_context - Handle long sequences
17. 16_synthetic_data - Generate training data

---

## Philosophy

> "Don't get stuck in theory. Code, debug, ablate, visualize. Your future self will thank you."

Each project:
1. **Code**: Hand-wire the concept from scratch
2. **Plot**: Visualize how it works
3. **Break**: Ablate components to understand impact
4. **Learn**: Extract the key insight

---

## Key Insights by Domain

### Model Architecture
- Attention is just weighted average of values
- Residuals enable deep networks
- Multiple heads: diversity principle
- Normalization prevents training collapse

### Inference
- KV cache is crucial for generation
- Batching is more important than raw speed
- Quantization: 4-bit is nearly lossless
- Framework choice: vLLM wins for production

### Scaling
- Loss improves ~O(1/N^0.07) with model size
- Compute budget determines optimal model size
- Larger models are more sample-efficient
- Chinchilla: balance model and data compute

### Training
- Choose objective based on downstream task
- Instruction tuning > supervised finetuning
- RLHF adds significant cost but better alignment
- Synthetic data can be as good as real data

---

## Advanced Topics (Future Additions)

- [ ] Sparse attention patterns
- [ ] Mixture of depths
- [ ] State space models (Mamba, S4)
- [ ] Continual learning
- [ ] Interpretability and mechanistic analysis
- [ ] Multi-modal (vision + language)
- [ ] Constitutional AI and process rewards

---

## References

- Attention Is All You Need (Vaswani et al., 2017)
- Scaling Laws for Neural Language Models (Kaplan et al., 2020)
- Chinchilla Scaling Laws (Hoffmann et al., 2022)
- Training Compute-Optimal LLMs (Hoffmann et al., 2022)
- RoPE: Rotary Position Embedding (Su et al., 2021)
- ALiBi: Attention with Linear Biases (Press et al., 2022)
- Switch Transformers: Scaling with Sparse Mixture of Experts
- Grouped Query Attention (Ainslie et al., 2023)

---

## Contributing

Found a bug? Want to add a project? Issues and PRs welcome!

---

## License

MIT

---

**Remember:** The best way to understand LLMs is to build them. Start coding.
