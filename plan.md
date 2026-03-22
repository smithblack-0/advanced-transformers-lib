# Implementation Plan: advanced-transformers-lib — Llama 3 Baseline

## Status
**Current state:** Units 1 and 2 verified. Unit 3 (mlp.py) in progress.

---

## Governing Principles

This plan records not just *what* to build but *why* each decision was made, so future sessions can
make good changes without reconstructing lost context. Every design decision includes its rationale.
Every component includes the invariants its tests must enforce. If an invariant changes, the tests must
change to match before the unit can be considered verified.

The architecture is a **synthesis**, not a transcription. Reference sources condition the implementation
but do not dictate its structure, style, or organisation. Where sources conflict or leave choices open,
a decision is made here and flagged for user review. Decisions with no reported rationale in the paper
are not replicated.

---

## Process Rules

These are non-negotiable. Deviating without explicit user approval is not acceptable.

**One unit at a time.** Never begin a new unit while the current one is unverified. A unit is not
complete until its tests pass and accurately describe the intended behaviour.

**Blocker stack.** When completing a unit requires work outside its scope (e.g. a missing config
parameter), that work is a new unit pushed onto a stack. Complete and verify the blocker, then return.
Only make the changes needed for compatibility — do not expand scope.

**Surface decisions.** Autonomous decisions are permitted but must be reported to the user for review.
Uncertain decisions must be escalated before proceeding. Do not resolve ambiguity silently.

**Keep this plan current.** Update the status section and unit checklist continuously. The plan must
reflect actual state at all times so work can be resumed after a session break without loss.

**Plan first within each unit.** At the start of each unit, confirm file granularity, list what will
be implemented, and identify any decision points before writing code.

---

## Code Quality Standards

These apply without exception to every file. Clean, well-structured code is a first-class requirement,
not a means to an end. Code that cannot be trusted is wrong by definition.

**Structure:**
- All architectural parameters expressed through config — no literal numbers that belong in config
- One responsibility per file — confirmed at the start of each unit
- Type hints on all function and method signatures
- No dead code
- Placeholders raise `NotImplementedError`, never pass silently
- Use PyTorch and library builtins wherever they exist — do not reimplement what is already provided

**Documentation:**
- All classes must have docstrings
- All public methods must have docstrings
- Private methods must have docstrings when the logic is not a single clear operation
- Document at the block level: what the block achieves and why this approach was chosen
- Do not narrate line by line what the code does — that is not documentation
- Skipping documentation on non-self-documenting code and writing useless line-by-line narration are
  failure modes of equal severity
- Code should be self-documenting through clear naming wherever possible; comments fill the gap where
  naming alone is insufficient

---

## Testing Philosophy

A codebase that works but cannot be verified has no value for research. Only a verified implementation
can be used to draw scientific conclusions. **Verified-but-imperfect is more valuable than
working-but-unverified** — only the former can be trusted as a research baseline.

Tests are first-class artifacts. They are written alongside the implementation, not appended afterward.
A component without passing tests is not complete regardless of how correct it appears.

**Rules:**
- Each src file has a corresponding test file mirroring the structure under `tests/llama3/`
- Unit tests verify each component in isolation
- Integration tests verify that combinations of components work together — they do not replicate the
  detail of unit tests
- Placeholders may fail, but must raise `NotImplementedError`
- When a unit is modified as a blocker, its tests must be updated to reflect the new correct behaviour
  before the unit is re-verified. Passing tests that no longer reflect intent are false confidence.
- **A bad test is worse than no test.** When uncertain how to correctly test a component, ask the user
  before writing the test.

---

## Repository Structure

```
advanced-transformers-lib/
├── src/
│   └── llama3/
│       ├── __init__.py
│       ├── configuration.py      # Unit 1
│       ├── rope.py               # Unit 2
│       ├── mlp.py                # Unit 3
│       ├── attention.py          # Unit 4
│       ├── decoder_layer.py      # Unit 5
│       ├── model.py              # Unit 6
│       └── upload_to_hub.py      # Unit 7
└── tests/
    └── llama3/
        ├── __init__.py
        ├── test_configuration.py
        ├── test_rope.py
        ├── test_mlp.py
        ├── test_attention.py
        ├── test_decoder_layer.py
        └── test_model.py
```

**Why this granularity:** One file per major responsibility. Each file has a clear, independent reason
to exist. The decoder layer and model are separate because a decoder layer can be verified in isolation
before being composed into a full model stack. `upload_to_hub.py` is separate because it is an
operational concern, not architecture.

**RMSNorm:** `torch.nn.RMSNorm` is used directly wherever normalisation is needed. There is nothing to
implement. At each point of use, a comment explains why RMSNorm was chosen over LayerNorm: it omits
mean subtraction, is faster, and proved more stable at scale. No separate file or unit.

File granularity is confirmed at the start of each unit per the process spec. The layout above is the
expected outcome but may be revised if a responsibility turns out to be smaller or larger than
anticipated.

---

## Implementation Order

The order below is preferred, not fixed. In practice, later units may surface gaps in earlier ones —
these are blockers and are handled via the stack mechanism in the process spec: push the current unit,
complete and verify the blocker, then return. Only the changes needed to resolve the blocker are made;
scope does not expand.

1. **configuration.py** — no dependencies
2. **rope.py** — depends on config for theta and rope_scaling
3. **mlp.py** — depends on config for sizes and bias flag
4. **attention.py** — depends on config and rope
5. **decoder_layer.py** — depends on attention, mlp, and torch.nn.RMSNorm
6. **model.py** — depends on all of the above; satisfies the HF AutoClass contract
7. **upload_to_hub.py** — depends on model and config being complete

---

## HuggingFace AutoClass Protocol

Understanding this protocol is prerequisite to Units 1, 6, and 7. It explains why certain things must
exist in the config and what the upload script is actually doing.

**How trust_remote_code works:** When a researcher calls
`AutoConfig.from_pretrained("namespace/repo", trust_remote_code=True)`, HuggingFace downloads the
Python files from the Hub repository and imports them in the local process. The `auto_map` field in
`config.json` tells HuggingFace which class in which file to instantiate. For example:

```json
"auto_map": {
  "AutoConfig": "configuration.Llama3Config",
  "AutoModelForCausalLM": "model.Llama3ForCausalLM"
}
```

This is why `model_type` must be unique — it is the key HuggingFace uses to look up the class in its
registry, and a collision with a built-in type would cause the wrong class to be loaded.

**What the upload script must push:** The Hub repository must contain the Python source files
(`configuration.py`, `rope.py`, `mlp.py`, `attention.py`, `decoder_layer.py`, `model.py`) alongside
`config.json`. HuggingFace downloads and executes these files on the researcher's machine.

**From-config instantiation flow:**
```python
config = AutoConfig.from_pretrained("namespace/repo", trust_remote_code=True)
model = AutoModelForCausalLM.from_config(config)  # fresh random weights, no checkpoint
```

---

## Units of Work

---

### Unit 1 — configuration.py

**What:** `Llama3Config`, a `PretrainedConfig` subclass. Every architectural parameter that varies
across model scales or is a meaningful research variable is expressed here. Constants that are fixed
architectural decisions of Llama 3 (no attention bias, SwiGLU activation, no tied embeddings) are
not parameters — they are implemented directly in the relevant module and documented at that point.

**Why it exists:** The HF AutoClass contract requires a config class with `model_type` set. Beyond
the contract, making every scale-variable parameter configurable is what allows this library to express
any model scale without touching implementation code.

**Parameters and their rationale:**

| Parameter | Type | Rationale |
|-----------|------|-----------|
| `vocab_size` | int | Embedding table rows and logit output dimension. Must match the tokenizer. |
| `hidden_size` | int | Model width. Every other dimension is either equal to this or a direct multiple/fraction of it. |
| `intermediate_size` | int | FFN width. **Direct parameter, not formula-derived.** Llama 3 ratios vary by scale (~3.5× at 8B/70B, ~3.25× at 405B). Computing from a formula would be wrong for some scales. |
| `num_hidden_layers` | int | Transformer stack depth. |
| `num_attention_heads` | int | Number of query heads. Determines how hidden_size is split per head. |
| `num_key_value_heads` | int | Number of KV heads (GQA). Must evenly divide `num_attention_heads`. Equal to `num_attention_heads` gives MHA; 1 gives MQA; values between give GQA. |
| `head_dim` | int \| None | Per-head dimension. Normally `hidden_size // num_attention_heads`. Exposed as a parameter because some architectures decouple head count from head size. Computed in `__post_init__` if None. |
| `rms_norm_eps` | float | Stability epsilon passed to `torch.nn.RMSNorm`. Prevents division by zero when activations are near zero. |
| `rope_theta` | float | Base rotation frequency for RoPE. Controls how fast angles rotate with position — higher means slower rotation, preventing aliasing at long distances. Llama 3 uses 500,000 to support up to 128K context. **This value carries physical meaning tied to the target context length and must never be hardcoded.** |
| `max_position_embeddings` | int | The context length the model was trained at. Required by HF's rope system as `original_max_position_embeddings` for scaling types (yarn, longrope, llama3). Llama 3 base: 8192. **Note:** this is the training context length, not an inference ceiling — the rope.py module handles longer sequences at runtime. |
| `rope_scaling` | dict \| None | Optional RoPE scaling for extending context beyond `max_position_embeddings`. Passed through to HF's `RotaryEmbeddingConfigMixin` which owns validation. HF format uses `rope_type` (not `type`) as the key. Supported types via HF's `ROPE_INIT_FUNCTIONS`: `"linear"`, `"dynamic"`, `"yarn"`, `"longrope"`, `"llama3"`. None means no scaling. |
| `attention_dropout` | float | Dropout on attention weights. Default 0.0. |
| `use_cache` | bool | Whether to return and accept `past_key_values`. True for inference, may be False during training to save memory. |

**Constants (not in config):** `attention_bias=False`, `mlp_bias=False` — fixed Llama 3 architectural
decisions with no reported rationale for variation. `hidden_act="silu"` — SwiGLU's gate uses SiLU
specifically; varying this changes the architecture in a way that belongs in a new variant, not a
config flag. These are documented as constants at the point of use.

**auto_map:** Config must include `auto_map` pointing `AutoConfig` and `AutoModelForCausalLM` to
the correct class paths. This enables `trust_remote_code` to find the right classes.

**model_type:** `"llama3_baseline"` — unique string that will not collide with HF's built-in `"llama"`.

**Weight initialisation:** We do not override `_init_weights`. PyTorch's defaults stand. `post_init()`
is called (required for HF gradient checkpointing machinery) but does not re-initialise weights.
Rationale: the paper reports no specific initialisation scheme, so replicating HF's normal-distribution
override would be adding something with no evidential basis in the synthesis.

**Invariants that must hold (tested):**
- `hidden_size % num_attention_heads == 0`
- `num_attention_heads % num_key_value_heads == 0`
- Serialisation round-trip: `Llama3Config.from_dict(config.to_dict())` produces an identical config
- Default instantiation succeeds without arguments
- Parameter overrides via kwargs work correctly (HF contract)

---

### Unit 2 — rope.py

**What:** `Llama3RotaryEmbedding` — computes and applies Rotary Position Embeddings to query and key
tensors.

**Why RoPE:** RoPE encodes position in the *relationship* between Q and K rather than adding it to the
values. When the attention dot product Q·Kᵀ is computed, the rotations cancel to produce a score
depending on the *relative* distance between positions. This gives better length generalisation than
absolute learned embeddings and more natural integration with attention than additive methods.

**How it works:** Each pair of head dimensions (d, d+1) is assigned a rotation frequency
`1 / theta^(2d / head_dim)`. Higher theta → slower rotation → encodings that distinguish positions
further apart before aliasing. At each forward pass, Q and K vectors are rotated by multiplying
dimension pairs by the corresponding cos/sin values for their positions.

**rope_theta:** The value 500,000 in Llama 3 is not arbitrary — Xiong et al. (2023) showed it is
effective up to 32,768 context as a prerequisite for long-context continued pretraining. Different
context length targets require different theta values. It must come from config.

**HF rope system:** Transformers 5.x owns rope configuration at the `PretrainedConfig` level via
`RotaryEmbeddingConfigMixin`. We do not fight this — `rope_scaling` and `rope_theta` are passed
through to HF's base class which validates and standardises them into `config.rope_parameters`.

**What rope.py provides:** HF's utilities compute the inverse frequencies (`ROPE_INIT_FUNCTIONS`)
but do not provide an `nn.Module`. `rope.py` implements `RotaryEmbedding`, which:
1. Computes inv_freq from config. Supported rope types: `"default"` (standard unscaled RoPE,
   computed directly), `"linear"` and `"yarn"` (delegated to `ROPE_INIT_FUNCTIONS`). All other
   types raise `NotImplementedError` — they can be added when needed.
2. Computes the cos/sin table from those frequencies.
3. Lazily extends the table when a sequence longer than the current cache is encountered — the table
   grows at runtime, making context length a runtime property rather than a build-time constraint.
   `max_position_embeddings` in config records the training context length (needed by HF's scaling
   computations as `original_max_position_embeddings`) but does not cap inference length.
4. Applies the rotation to Q and K tensors.

The table is registered as a buffer so it moves with the model to the correct device.

**Invariants that must hold (tested):**
- `output.shape == input.shape` — rotation is shape-preserving
- At position 0: rotation is identity — output equals input (cos=1, sin=0)
- Relative position property: `dot(RoPE(q, i), RoPE(k, j))` depends only on `(i - j)`. Verified by
  checking that two different absolute position pairs with the same relative offset produce the same
  dot product. This is the core correctness guarantee of RoPE.
- `rope_theta` affects the frequency table (changing it produces different output)
- Lazy extension: processing a sequence longer than the initial table extends correctly and subsequent
  forward passes produce consistent results

---

### Unit 3 — mlp.py

**What:** `SwiGLUMLP` — the feed-forward sublayer using SwiGLU activation.

**Why SwiGLU over ReLU/GeLU:** SwiGLU is a gated linear unit variant that applies SiLU as a gate
multiplied element-wise against a separate linear projection. The gating mechanism gives the network
more expressive control over which features to propagate. At this model generation it empirically
outperforms both ReLU and GeLU variants. It requires three weight matrices (gate, up, down) instead
of two, which is why `intermediate_size` in Llama 3 is lower than the 4× multiplier typical of
two-matrix FFNs — the total parameter count is comparable.

**Formula:** `output = W_down(SiLU(W_gate(x)) ⊙ W_up(x))`

All three projections are `nn.Linear` with `bias=False`:
- `W_gate`: `nn.Linear(hidden_size, intermediate_size, bias=False)`
- `W_up`:   `nn.Linear(hidden_size, intermediate_size, bias=False)`
- `W_down`: `nn.Linear(intermediate_size, hidden_size, bias=False)`

No bias on any projection — fixed architectural choice with no rationale for variation.

SiLU (not ReLU, not GeLU) is used as the gate activation because SwiGLU specifically refers to this
combination. Applied via `torch.nn.functional.silu` — no custom implementation. It is not a config
parameter — varying the gate activation produces a different architecture, not a different scale of
this one.

**Typing convention:** `config` is typed as `PretrainedConfig` (not `Llama3Config`), consistent
with `RotaryEmbedding`. The module reads only `hidden_size` and `intermediate_size`, both of which
are standard config attributes — no reason to couple the type to our specific subclass.

**Invariants that must hold (tested):**
- Input `(batch, seq, hidden_size)` → output `(batch, seq, hidden_size)`
- No bias on any projection
- Gating is active: zeroing `W_gate` output zeros the final output (the gate controls the signal)

---

### Unit 4 — attention.py

**What:** `Llama3Attention` — Grouped Query Attention with causal masking and KV cache support.

**Why GQA:** At 128K context, the KV cache dominates memory. With 8 KV heads and 32 query heads (8B
model), the KV cache is 4× smaller than standard MHA. This was the primary architectural motivation —
GQA makes 128K context practical. Llama 3 uses 8 KV heads at all scales (8B, 70B, 405B). The
implementation must support arbitrary `num_key_value_heads` because this is the primary research
variable this parameter enables.

**Why KV cache:** Because GQA exists to serve KV caching, caching must be implemented. It is the
raison d'être of the design choice.

**Projections:**
- Q: `hidden_size → num_attention_heads × head_dim`, no bias
- K: `hidden_size → num_key_value_heads × head_dim`, no bias
- V: `hidden_size → num_key_value_heads × head_dim`, no bias
- O: `num_attention_heads × head_dim → hidden_size`, no bias

No bias on any projection — Llama 3 architectural constant.

**KV head expansion:** Before computing attention, K and V are repeated
`num_attention_heads // num_key_value_heads` times along the head dimension to align with Q. This is
what makes GQA compatible with standard multi-head attention computation.

**Attention kernel:** `torch.nn.functional.scaled_dot_product_attention` (SDPA). PyTorch's unified
kernel automatically selects FlashAttention when hardware and dtype support it, falling back to
standard attention otherwise. This delivers efficiency without an additional dependency and uses a
well-tested PyTorch builtin rather than a custom kernel.

**Causal masking:** `is_causal=True` passed to SDPA during prefill (processing the full input
sequence). During cached generation (single new token attending over full cached history), no mask is
needed — the new token is always the last position.

**KV cache format:** Tuple of `(key_states, value_states)` per layer, passed as `past_key_values`.
Classic HF format, broadly compatible. On each forward pass, new K/V are concatenated with cached
K/V along the sequence dimension before attention is computed.

**Invariants that must hold (tested):**
- Output shape matches input shape
- `num_attention_heads % num_key_value_heads == 0` — checked defensively at runtime
- GQA correctness: with `num_key_value_heads < num_attention_heads`, output shape is correct and KV
  expansion is occurring (verified by inspecting expanded tensor shapes)
- Causal masking: attention weight matrix is lower-triangular (token i attends only to positions ≤ i)
- KV cache: output at position t with cache equals output at position t from a full forward pass over
  the complete sequence up to t
- No bias on any projection

---

### Unit 5 — decoder_layer.py

**What:** `Llama3DecoderLayer` — a single transformer block: pre-norm attention followed by pre-norm
MLP, with residual connections.

**Why pre-norm:** Normalising the sublayer *input* (not output) keeps the residual stream
unnormalised. Gradients flow more cleanly through unnormalised residuals at depth, and each sublayer
operates on a stable, normalised view of the signal. This is why large language models trained at
scale use pre-norm.

**Structure:**
```
h   = x + Attention(RMSNorm(x), ...)
out = h + MLP(RMSNorm(h))
```

Two independent `torch.nn.RMSNorm` instances — one before attention, one before MLP. They learn
different scalings because they precede layers with different dynamic ranges. This is not an
implementation detail; sharing them would be wrong.

**Invariants that must hold (tested):**
- Input and output shapes are identical
- Two independent RMSNorm parameter tensors (not the same object)
- Residual connections are present: bypassing them changes the output
- Integration: attention output feeds correctly into the MLP residual path

---

### Unit 6 — model.py

**What:** `Llama3Model` (transformer backbone) and `Llama3ForCausalLM` (backbone + LM head).
This unit satisfies the HuggingFace AutoClass contract.

**Why two classes:** `Llama3Model` is the pure transformer backbone, reusable for any task head.
`Llama3ForCausalLM` adds the language modelling head and loss computation. This separation matches HF
conventions and is the correct division of responsibility.

**Llama3Model:**
- Token embedding: `vocab_size × hidden_size`
- Stack of `num_hidden_layers` decoder layers
- Final `torch.nn.RMSNorm` — the stack output is normalised before projection to logits
- Returns last hidden state; optionally all hidden states, attention weights, past_key_values

**Llama3ForCausalLM:**
- Contains `Llama3Model` as `self.model`
- LM head: `nn.Linear(hidden_size, vocab_size, bias=False)`
- `tie_word_embeddings`: if True, LM head weight is shared with the embedding table
- `forward()`: runs model, projects to logits, computes cross-entropy loss if labels provided
  (labels are shifted by one — each token predicts the next)
- Returns `CausalLMOutputWithPast`

**HF contract:**
- Inherits `PreTrainedModel` and `GenerationMixin`
- `config_class = Llama3Config`
- `base_model_prefix = "model"`
- `_no_split_modules = ["Llama3DecoderLayer"]`
- `supports_gradient_checkpointing = True`
- `post_init()` called at end of `__init__` (required for HF gradient checkpointing machinery)
- `_init_weights` is NOT overridden — PyTorch default initialisation stands. The paper reports no
  specific initialisation scheme; replicating HF's normal-distribution override would add something
  with no basis in the synthesis.

**Invariants that must hold (tested):**
- Output logits shape: `(batch, seq, vocab_size)`
- Loss computed correctly when labels provided (cross-entropy, next-token prediction)
- KV cache: generation with cache produces identical logits to full forward at the current position
- `save_pretrained` / `from_pretrained` round-trip: all weights identical
- `AutoModelForCausalLM.from_config(config)` instantiates without error

---

### Unit 7 — upload_to_hub.py

**What:** Standalone script that makes the architecture available on HuggingFace Hub so a researcher
can instantiate a fresh model with no checkpoint.

**Why separate:** Operational concern, not architecture. No model code imports this. It runs once per
release.

**Responsibilities:**
- Accept the Hub repository name as a command-line argument — never hardcoded
- Register `Llama3Config` with `AutoConfig` and `Llama3ForCausalLM` with `AutoModelForCausalLM`
- Push all src files (`configuration.py`, `rope.py`, `mlp.py`, `attention.py`, `decoder_layer.py`,
  `model.py`) to the Hub repository alongside `config.json`
- Upload the Llama 3 tokenizer to the same repository
- Generate and push a model card populated with architectural details. **Whether HuggingFace supports
  an architecture-only card (without concrete trained weights) is unknown and must be investigated
  during this unit.** The model card is generated programmatically — architectural details are stored
  as data, not inline strings.
- Never upload weights. The script must make it structurally impossible to accidentally upload a
  checkpoint.

**Testing:** Hub interaction cannot be unit tested without a live connection. Manual verification is
appropriate: run against a test namespace, then confirm that the from-config instantiation flow works
from a fresh environment.

---

## Open Decisions

The following decisions were made during planning and are flagged for confirmation. They will be
revisited at the start of the relevant unit.

| # | Decision | Proposal | Status |
|---|----------|----------|--------|
| 1 | `model_type` | `"llama3_baseline"` | Confirmed by user |
| 2 | Attention kernel | `torch.nn.functional.scaled_dot_product_attention` | Confirmed by user |
| 3 | KV cache format | Tuple of `(key, value)` per layer | Confirmed by user |
| 4 | YaRN scaling | Handled natively by HF's `ROPE_INIT_FUNCTIONS` — no placeholder needed | Resolved: all scaling types supported via HF |
| 5 | `intermediate_size` | Direct config parameter | Confirmed by user |

---

## Session Resume Instructions

Read this file first on session start. The status section at the top reflects actual current state.
The checklist below is the ground truth for progress — unchecked means not started, in-progress means
started but not yet verified, checked means verified (tests passing).

Do not begin a new unit while the current one is unverified. If a blocker arises mid-unit, push the
current unit, resolve and verify the blocker, then return.

- [x] Unit 1 — configuration.py
- [x] Unit 2 — rope.py
- [ ] Unit 3 — mlp.py  ← in progress
- [ ] Unit 3 — mlp.py
- [ ] Unit 4 — attention.py
- [ ] Unit 5 — decoder_layer.py
- [ ] Unit 6 — model.py
- [ ] Unit 7 — upload_to_hub.py
