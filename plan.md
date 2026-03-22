# Implementation Plan: advanced-transformers-lib — Llama 3 Baseline

## Status
**Current state:** Planning — awaiting user approval before any code is written.

---

## Governing Principles

This plan exists to record not just *what* to build but *why* each decision was made, so that future
sessions can make good changes without reconstructing lost context. Every design decision includes its
rationale. Every component includes the invariants that must hold when it is verified. These invariants
are what the tests enforce — if an invariant changes, the tests must change to match before the unit
can be considered verified.

The architecture is a **synthesis**, not a transcription. The reference sources condition the
implementation but do not dictate its structure, style, or organisation. Where sources conflict or leave
choices open, a decision is made here and flagged for user review.

---

## Repository Structure

```
advanced-transformers-lib/
├── src/
│   └── llama3/
│       ├── __init__.py
│       ├── configuration.py      # Unit 1
│       ├── norm.py               # Unit 2
│       ├── rope.py               # Unit 3
│       ├── mlp.py                # Unit 4
│       ├── attention.py          # Unit 5
│       ├── decoder_layer.py      # Unit 6
│       ├── model.py              # Unit 7
│       └── upload_to_hub.py      # Unit 8
└── tests/
    └── llama3/
        ├── __init__.py
        ├── test_configuration.py
        ├── test_norm.py
        ├── test_rope.py
        ├── test_mlp.py
        ├── test_attention.py
        ├── test_decoder_layer.py
        └── test_model.py
```

**Why this granularity:** One file per major responsibility. Each file has a clear, independent reason
to exist. The decoder layer and model are separate because a decoder layer can be verified in isolation
before being composed into a full model stack — that separation is what makes the tests meaningful.
`upload_to_hub.py` is separate because it is an operational concern, not architecture.

File granularity within each unit is confirmed at the start of that unit per the process spec. The above
layout is the expected outcome but may be revised if a responsibility turns out to be smaller or larger
than anticipated.

---

## Implementation Order

Bottom-up. Each unit depends only on previously verified units, so each becomes a trusted black box
before the next unit is written.

1. **configuration.py** — no dependencies
2. **norm.py** — depends on config for eps parameter
3. **rope.py** — depends on config for theta, max_position_embeddings, rope_scaling
4. **mlp.py** — depends on config for sizes and bias flags
5. **attention.py** — depends on config, rope, norm (for optional qk norm)
6. **decoder_layer.py** — depends on attention, mlp, norm
7. **model.py** — depends on all of the above; satisfies the HF contract
8. **upload_to_hub.py** — depends on model and config being complete

---

## Units of Work

---

### Unit 1 — configuration.py

**What:** `Llama3Config`, a `PretrainedConfig` subclass. Defines every architectural parameter as a
typed, documented field with a sensible default. This is the single source of truth for all model
dimensions — nothing in the architecture may use a literal number that belongs in the config.

**Why it exists:** The HF AutoClass contract requires a config class. More importantly, making every
parameter configurable is what allows this library to express any model scale (8B, 70B, 405B, or a
researcher's custom variant) without touching implementation code. The moment a number is hardcoded, the
library has made an assumption that breaks for every other scale.

**Parameters and their rationale:**

| Parameter | Type | Rationale |
|-----------|------|-----------|
| `vocab_size` | int | Embedding table rows and logit output dimension. Must match tokenizer. |
| `hidden_size` | int | Model width. Every other dimension is either equal to this or a multiple/fraction of it. |
| `intermediate_size` | int | FFN width. **Direct parameter, not derived.** Llama 3 does not use a fixed ratio to hidden_size (ratios vary: ~3.5× at 8B/70B, ~3.25× at 405B). Computing it from a formula would bake in the wrong assumption for some scales. |
| `num_hidden_layers` | int | Transformer stack depth. |
| `num_attention_heads` | int | Number of query heads. Determines how the hidden dimension is split per head. |
| `num_key_value_heads` | int | Number of KV heads (GQA). Setting equal to `num_attention_heads` gives standard MHA; setting to 1 gives MQA; values in between give GQA. Must evenly divide `num_attention_heads`. |
| `head_dim` | int \| None | Per-head dimension. Normally `hidden_size // num_attention_heads`. Exposed as a parameter because some architectures decouple head count from head size. Computed in `__post_init__` if None. |
| `rms_norm_eps` | float | Stability epsilon for RMSNorm. Prevents division by zero when a layer's activations are near zero. |
| `rope_theta` | float | Base rotation frequency for RoPE. Controls how fast angles rotate with position — higher theta means slower rotation, preventing aliasing at long distances. Llama 3 uses 500,000 (vs ~10,000 typical) to support up to 128K context. **Must come from config; the specific value carries physical meaning and varies with target context length.** |
| `rope_scaling` | dict \| None | Optional RoPE scaling config for extending context at inference without retraining. Keys: `type` (e.g. `"linear"`, `"yarn"`), `factor`, and type-specific parameters. See rope.py for supported types. None means no scaling applied. |
| `max_position_embeddings` | int | Maximum sequence length. Used to precompute the RoPE frequency table. Must be set to at least the longest sequence the model will process. |
| `attention_bias` | bool | Whether Q/K/V/O projections include a bias term. False for Llama 3 (inherited from Llama 2 — empirically better without). |
| `mlp_bias` | bool | Whether MLP projections include a bias term. False for Llama 3 (same reasoning). |
| `attention_dropout` | float | Dropout probability on attention weights. Default 0.0 for deterministic inference. |
| `hidden_act` | str | Activation function name for the MLP gate. `"silu"` for SwiGLU. Expressed as a string so it can vary across model variants without code changes. |
| `initializer_range` | float | Standard deviation for normal weight initialisation. |
| `use_cache` | bool | Whether to return and accept `past_key_values` for KV caching. Should be True for inference, may be False during training to save memory. |
| `tie_word_embeddings` | bool | Whether the input embedding and LM head share weights. False for Llama 3. |

**auto_map:** The config must include an `auto_map` dict pointing `AutoConfig` and
`AutoModelForCausalLM` to the correct class paths in this module. This is what makes `trust_remote_code`
work from the Hub.

**Invariants that must hold (tested):**
- `hidden_size % num_attention_heads == 0` — the hidden dimension must split evenly across heads
- `num_attention_heads % num_key_value_heads == 0` — query heads must split evenly across KV head groups
- Serialisation round-trip: `Llama3Config.from_dict(config.to_dict())` produces an identical config
- All defaults produce a valid config without raising
- Parameter overrides via kwargs at instantiation work correctly (HF contract)

**Decision flagged for user review:**
- `model_type` string — must be unique to avoid colliding with HF's built-in `"llama"` model. Proposal:
  `"llama3"`. If this causes any HF compatibility issue it can be changed without affecting logic.

---

### Unit 2 — norm.py

**What:** `RMSNorm` — Root Mean Square Layer Normalization.

**Why RMSNorm over LayerNorm:** LayerNorm subtracts the mean then divides by std. RMSNorm skips the mean
subtraction and divides by the root mean square only. This is faster (one fewer pass over the tensor)
and empirically more stable for large transformer training. The learned weight (gamma) still rescales
the output. Llama 3 inherits this from Llama 2 where it was validated at scale.

**Why pre-norm:** Normalisation is applied to the input of each sublayer (attention, MLP), not the
output. Pre-norm improves training stability by keeping the residual stream unnormalised — the gradients
flow more cleanly through residual connections.

**Implementation notes:**
- Input is upcast to float32 for the RMS computation even in mixed precision contexts. This prevents
  numerical instability in the normalisation itself. Output is then downcast back to the input dtype.
- The epsilon (`rms_norm_eps`) must come from config — it is not a universal constant.

**Invariants that must hold (tested):**
- `output.shape == input.shape` — normalisation is shape-preserving
- When `weight` is all-ones and input already has unit RMS, output ≈ input (up to floating point tolerance)
- Manual RMS computation matches the module output
- Near-zero input does not produce NaN or inf (epsilon is doing its job)
- Output dtype matches input dtype (float32 upcast is internal only)

---

### Unit 3 — rope.py

**What:** Rotary Position Embeddings (RoPE) — encodes sequence position by rotating query and key
vectors in 2D subspaces.

**Why RoPE:** RoPE encodes position in the *relationship* between Q and K vectors rather than adding it
to the values. When you compute the attention dot product Q·K^T, the positional rotations cancel to
produce a score that depends on the *relative* distance between positions. This gives RoPE better
length-generalisation than absolute learned embeddings and more natural integration with the attention
operation than additive methods like ALiBi.

**How it works:** Each head dimension pair (d, d+1) is assigned a rotation frequency
`1 / theta^(2d / head_dim)`. Higher theta → slower rotation → encodings that distinguish positions
further apart before aliasing. At inference, the precomputed cos/sin table is indexed by position,
and Q/K vectors are rotated by multiplying pairs of dimensions by the rotation matrix.

**rope_theta from config:** The value 500,000 in Llama 3 is not arbitrary — it was chosen to prevent
aliasing up to 32,768 tokens (Xiong et al. 2023) as a prerequisite for the long-context continued
pretraining. Different context targets require different theta values. It must come from config.

**rope_scaling:** When a model trained at context length L needs to operate at length L' > L, the
RoPE frequencies at positions > L are ones the model has never seen. Scaling techniques address this by
interpolating the frequencies to fit the extended range.

Supported types:
- `"linear"`: Divides all frequencies by `factor`. Effectively compresses positions into the trained
  range. Simple and effective for moderate extension. Formula: `freq_scaled = freq / factor`.
- `"yarn"`: Not Yet Another RoPE iNterpolation — applies different scaling to low- vs high-frequency
  dimensions (high-frequency dimensions rotate fast and are more sensitive to interpolation error).
  **Placeholder in this implementation — raises `NotImplementedError` with a clear message explaining
  what YaRN does and what is needed to implement it.** The config schema supports it so researchers
  can see it is planned.

**Implementation approach (synthesis decision, flagged):**
- Precompute a (max_position_embeddings, head_dim/2) cos/sin table at construction time.
- Apply to Q/K by rotating dimension pairs using the standard rotation formula.
- This is the author's own structure — not copied from any source.

**Invariants that must hold (tested):**
- `output.shape == input.shape` — rotation is shape-preserving
- At position 0: cos=1, sin=0 → rotation is identity → output equals input
- Relative position property: `dot(RoPE(q, i), RoPE(k, j))` depends only on `(i - j)`, not on
  absolute positions. This is the core correctness guarantee of RoPE. Verified by checking that
  shifted position pairs give the same dot product as the base pair.
- `rope_theta` affects output: doubling theta changes the frequency table (verified by comparison)
- Linear scaling: frequencies are divided by factor (verified against manual computation)
- YaRN: raises `NotImplementedError`

---

### Unit 4 — mlp.py

**What:** `Llama3MLP` — the feed-forward sublayer using SwiGLU activation.

**Why SwiGLU over ReLU/GeLU:** SwiGLU is a gated linear unit variant that applies SiLU (Sigmoid Linear
Unit, also called Swish) as a gate multiplied element-wise against a separate linear projection. This
gating mechanism gives the network more expressive control over which features to pass through. At this
model generation it empirically outperforms both ReLU and GeLU variants. It requires three weight
matrices (gate, up, down) instead of two, which is why `intermediate_size` in Llama 3 is set lower
than the 4× multiplier typical of two-matrix FFNs — the parameter count is similar.

**Formula:** `output = W_down(SiLU(W_gate(x)) ⊙ W_up(x))`

- `W_gate`: hidden_size → intermediate_size (no bias unless mlp_bias)
- `W_up`: hidden_size → intermediate_size (no bias unless mlp_bias)
- `W_down`: intermediate_size → hidden_size (no bias unless mlp_bias)

**Why intermediate_size is a direct parameter:** See configuration.py rationale. The formula-derived
value from Llama 2 (`int(2/3 * 4 * hidden_size)` rounded to multiple_of) does not match Llama 3's
actual dimensions at any scale. Using a direct parameter is correct and simpler.

**Invariants that must hold (tested):**
- Input shape `(batch, seq, hidden_size)` → output shape `(batch, seq, hidden_size)`
- No bias when `mlp_bias=False`; bias present when `mlp_bias=True`
- The gating mechanism is active: `SiLU(gate) * up` drives the output (verified by checking that
  zeroing the gate projection zeros the output)

---

### Unit 5 — attention.py

**What:** `Llama3Attention` — Grouped Query Attention with causal masking and KV cache support.

**Why GQA:** At 128K context length, the KV cache dominates memory. With standard MHA at 32 heads, the
KV cache is proportional to 2 × seq_len × num_heads × head_dim. With 8 KV heads, it is 4× smaller
(for the 8B model). GQA was the primary architectural change that made 128K context practical in Llama
3. The implementation must support this correctly — sharing K and V across groups of Q heads while
computing attention correctly.

**KV cache:** Because GQA exists to serve KV caching, caching must be implemented. At inference,
each decoder step appends the new K/V to the cache and attends over the full cached history. Without
this, the O(n²) attention cost at long context would defeat the point.

**Structure:**
- Q projection: `hidden_size → num_attention_heads × head_dim`
- K projection: `hidden_size → num_key_value_heads × head_dim`
- V projection: `hidden_size → num_key_value_heads × head_dim`
- O projection: `num_attention_heads × head_dim → hidden_size`
- KV expansion: before attention, K and V are repeated
  `num_attention_heads // num_key_value_heads` times along the head dimension to align with Q

**Attention computation (synthesis decision, flagged):**
Use `torch.nn.functional.scaled_dot_product_attention` (SDPA). This is PyTorch's unified attention
kernel that automatically selects FlashAttention when the hardware and dtype support it, and falls back
to standard attention otherwise. This is the author's synthesis decision — it is not copied from any
source, and it delivers efficiency without requiring a flash attention dependency.

**Causal masking:** Passed to SDPA as `is_causal=True` during prefill. During cached generation
(single new token attending over full cache), masking is unnecessary because the new token is always
the last position.

**KV cache format (decision flagged):** Use a simple tuple-of-tensors per layer
`(key_states, value_states)` passed as `past_key_values`. This is the classic HF format, broadly
compatible. Newer HF versions use `Cache` objects — this can be upgraded in a future session if needed.

**Invariants that must hold (tested):**
- Output shape matches input shape
- `num_attention_heads % num_key_value_heads == 0` — checked defensively at runtime (config should
  enforce this, but belt-and-suspenders here because a violation would produce a silent shape error)
- GQA correctness: with `num_key_value_heads < num_attention_heads`, output is still correct
  (verified by checking that a full-MHA config and a GQA config with the same effective capacity
  produce outputs of the correct shape and that KV expansion is happening)
- Causal masking: token at position i produces logits influenced only by positions ≤ i. Verified by
  checking that the attention weight matrix is lower-triangular.
- KV cache: output at position t with cache equals output at position t from a full forward pass
  over the complete sequence up to t
- No bias when `attention_bias=False`

---

### Unit 6 — decoder_layer.py

**What:** `Llama3DecoderLayer` — a single transformer block: pre-norm attention + pre-norm MLP with
residual connections.

**Why this structure:** Pre-norm residual architecture is the standard for large language models
trained at scale. The residual connections ensure gradient flow through the full depth of the model.
Pre-norm (normalise the sublayer input, not output) keeps the residual stream clean — the unnormalised
residual carries information across layers while each sublayer operates on a normalised view of it.

**Structure:**
```
h = x + Attention(RMSNorm_attn(x), ...)
out = h + MLP(RMSNorm_mlp(h))
```

Two separate RMSNorm instances with independent parameters — one for attention input, one for MLP
input. This is not an implementation detail; the two norms learn different scaling because they
precede layers with different dynamic ranges.

**Invariants that must hold (tested):**
- Input and output shapes are identical
- Two independent RMSNorm parameter sets (verified by checking they have different parameter tensors)
- Residual connections are present: removing them changes the output (not just a pass-through)
- Integration: the attention and MLP sublayers interact correctly through the residual stream

---

### Unit 7 — model.py

**What:** `Llama3Model` (transformer backbone) and `Llama3ForCausalLM` (backbone + LM head). This unit
satisfies the HuggingFace AutoClass contract.

**Why split into two classes:** `Llama3Model` is the pure transformer — it can be reused as a backbone
for any task (sequence classification, token classification, etc.). `Llama3ForCausalLM` adds the
language modelling head and the loss computation. This matches HF conventions and is the correct
separation of concerns.

**Llama3Model:**
- Token embedding table: `vocab_size × hidden_size`
- Stack of `num_hidden_layers` decoder layers
- Final RMSNorm (the output of the stack is normalised before being projected to logits)
- Returns: last hidden state, optionally all hidden states, optionally all attention weights,
  optionally past_key_values

**Llama3ForCausalLM:**
- Contains `Llama3Model` as `self.model`
- LM head: `nn.Linear(hidden_size, vocab_size, bias=False)`
- If `tie_word_embeddings=True`: LM head weight is shared with the embedding table
- `forward()`: runs model, projects to logits, computes cross-entropy loss if labels provided
  (labels shifted by one — each token predicts the next)
- Returns `CausalLMOutputWithPast`

**HF contract requirements:**
- Inherits `PreTrainedModel` and `GenerationMixin`
- `config_class = Llama3Config`
- `base_model_prefix = "model"`
- `_no_split_modules = ["Llama3DecoderLayer"]`
- `supports_gradient_checkpointing = True`
- `post_init()` called at end of `__init__`
- `auto_map` in config points to this class

**Invariants that must hold (tested):**
- Output logits shape: `(batch, seq, vocab_size)`
- Loss is computed correctly when labels provided (cross-entropy on shifted sequence)
- KV cache: generation with cache produces identical logits to full forward for the current position
- `save_pretrained` / `from_pretrained` round-trip: all weights preserved exactly
- `AutoModelForCausalLM.from_config(config)` instantiates without error (AutoClass contract)
- Gradient checkpointing: enabling it does not change forward pass output (only memory profile)

---

### Unit 8 — upload_to_hub.py

**What:** A standalone script that makes the architecture available on HuggingFace Hub so a researcher
can pull it and instantiate a fresh model with no checkpoint.

**Why separate:** This is operational, not architectural. It has no business being imported by any
model code. It runs once per release.

**Responsibilities:**
- Register `Llama3Config` with `AutoConfig` and `Llama3ForCausalLM` with `AutoModelForCausalLM`
- Push `configuration.py` and `model.py` (and all transitive src files) to the Hub repository
- Upload the Llama 3 tokenizer to the same Hub repository
- Generate and push a model card populated with architectural details (programmatically — large text
  stored as data, not inline strings)
- **Never upload weights.** The script must make it impossible to accidentally upload a checkpoint.

**Testing:** Hub interaction cannot be meaningfully unit tested without a live Hub connection. Manual
verification is the appropriate check: run the script against a test namespace, then verify that
`AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(..., trust_remote_code=True))` works
from a fresh Python environment.

---

## Open Decisions Requiring User Input

The following decisions were made autonomously during planning and are flagged for review before or
during implementation. They will be revisited at the start of the relevant unit.

| # | Decision | Proposal | Reason |
|---|----------|----------|--------|
| 1 | `model_type` string in config | `"llama3"` | Must not collide with HF's built-in `"llama"` model type |
| 2 | Attention kernel | `torch.nn.functional.scaled_dot_product_attention` | Automatic FlashAttention selection, no extra dependency |
| 3 | KV cache format | Tuple of `(key, value)` tensors per layer | Broad HF compatibility; can be upgraded to Cache objects later |
| 4 | RoPE scaling — YaRN | `NotImplementedError` placeholder | Complex to implement correctly; linear scaling covers the near-term use case |
| 5 | `intermediate_size` | Direct config parameter | Llama 3 ratios vary by scale; formula-derived value is wrong for some scales |

---

## Session Resume Instructions

On session start: read this file first. The status section reflects actual current state. Each unit has
a checkbox — unchecked means not started, in-progress means started but not verified, checked means
verified (tests passing and reviewed). Do not begin a new unit until the current one is checked.

Unit completion checklist:
- [ ] Unit 1 — configuration.py
- [ ] Unit 2 — norm.py
- [ ] Unit 3 — rope.py
- [ ] Unit 4 — mlp.py
- [ ] Unit 5 — attention.py
- [ ] Unit 6 — decoder_layer.py
- [ ] Unit 7 — model.py
- [ ] Unit 8 — upload_to_hub.py
