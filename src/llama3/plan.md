# Implementation Plan: advanced-transformers-lib — Llama 3 Baseline

## What This Document Is

This is the planning and process record for the LLM-assisted implementation of the Llama 3
baseline. It was the active working document throughout development — every unit was planned
here before code was written, every decision was recorded here for human review, and every unit
was verified here before the next began.

It exists for two reasons. First, as a development tool: it allowed work to resume across
sessions without loss of context. Second, and more importantly, as a trust artifact: it is
the evidence that this LLM-produced codebase was built under rigorous human supervision.

The process it records:

- Before writing any code for a unit, the plan for that unit was written and reviewed.
- Each unit was implemented, tested, and verified before the next began.
- When a blocker arose, it was pushed onto a stack, resolved as its own verified unit, and
  then the interrupted unit resumed. No blocker was hacked around inline.
- All significant decisions were surfaced for human review. The human author approved or
  corrected each one. Autonomous decisions were reported, not hidden.

The checklist below is the visible record that this process was followed for every unit.

## Completion Record

- [x] Unit 1 — configuration.py
- [x] Unit 2 — rope.py
- [x] Unit 3 — mlp.py
- [x] Unit 4 — attention.py
- [x] Unit 5 — decoder_layer.py
- [x] Blocker — type_aliases.py
- [x] Unit 6 — model.py (Llama3Model)
- [x] Unit 7 — huggingface.py (Llama3ForCausalLM)
- [x] Blocker — auto_map in configuration.py
- [x] Blocker A — embed_tokens belongs on Llama3ForCausalLM, not Llama3Model
- [x] Blocker B — override _init_weights to no-op so PyTorch constructor defaults stand
- [x] Blocker — Refactor: move files into src/llama3/model/, convert to relative imports
- [x] Blocker — tokenizer.py
- [x] Unit 8 — upload_to_hub.py
- [x] Unit 9 — Documentation
- [x] Blocker — forward() return types: plain dict → ModelOutput
- [x] Blocker — _reorder_cache for beam search
- [x] Blocker — Integration Tests
- [x] Blocker — DynamicCache compatibility
- [x] Blocker — model.py pure torch refactor
- [x] Unit 10 — End-to-End Tests
- [ ] Unit 11 — Audit

---

## Status
**Current state:** Units 1–10 verified. 114 local tests + 6 network tests pass. Unit 11 (Audit) next.

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
- Each src file has a corresponding test file mirroring dsfthe structure under `tests/llama3/`
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
│       ├── model/                  ← WYSIWYG: contents uploaded flat to Hub root
│       │   ├── __init__.py
│       │   ├── configuration.py    # Unit 1
│       │   ├── rope.py             # Unit 2
│       │   ├── mlp.py              # Unit 3
│       │   ├── attention.py        # Unit 4
│       │   ├── decoder_layer.py    # Unit 5
│       │   ├── model.py            # Unit 6
│       │   ├── huggingface.py      # Unit 7
│       │   ├── type_aliases.py     # Blocker
│       │   └── README.md           # Model card — pushed to Hub root as README.md
│       ├── upload_to_hub.py        # Unit 8
│       └── tokenizer.py            # Blocker
└── tests/
    └── llama3/
        ├── __init__.py
        ├── test_tokenizer.py           # mirrors src/llama3/tokenizer.py
        └── model/
            ├── __init__.py
            ├── test_configuration.py
            ├── test_rope.py
            ├── test_mlp.py
            ├── test_attention.py
            ├── test_decoder_layer.py
            ├── test_model.py
            └── test_huggingface.py
```

**Why this granularity:** One file per major responsibility. Each file has a clear, independent reason
to exist. The decoder layer and model are separate because a decoder layer can be verified in isolation
before being composed into a full model stack. `upload_to_hub.py` and `tokenizer.py` are separate
because they are operational concerns, not architecture. The `model/` subfolder is the WYSIWYG Hub
distribution unit — its contents are uploaded flat to the Hub repository root via `upload_folder`.

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

**What:** `GroupedQueryAttention` — Grouped Query Attention with causal masking and KV cache support.

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

**Causal masking:** Strictly causal — no external attention mask parameter. `is_causal=True` passed
to SDPA during prefill (q_len > 1). During cached generation (q_len == 1), `is_causal=False` —
the new token is always the last position and may attend to the full history without a mask.

**attention_scaling:** `RotaryEmbedding.forward` returns an `attention_scaling` factor (1.0 for
default/linear, != 1.0 for YaRN). This is applied by passing `scale=attention_scaling/sqrt(head_dim)`
to SDPA, which replaces the default `1/sqrt(head_dim)` scaling. This correctly adjusts attention
magnitude for scaling types that manipulate frequency structure.

**KV cache format:** Tuple of `(key_states, value_states)` per layer, passed as `past_key_values`.
Classic HF format, broadly compatible. On each forward pass, new K/V are concatenated with cached
K/V along the sequence dimension before attention is computed.

**Invariants that must hold (tested):**
- Output shape matches input shape
- `num_attention_heads % num_key_value_heads == 0` — checked defensively at runtime
- GQA correctness: with `num_key_value_heads < num_attention_heads`, output shape is correct and KV
  expansion is occurring (verified by inspecting expanded tensor shapes)
- Causal masking: future tokens do not influence past token outputs — verified by confirming that
  modifying tokens at positions > t leaves the output at position t unchanged
- KV cache: output at position t with cache equals output at position t from a full forward pass over
  the complete sequence up to t
- No bias on any projection

---

### Blocker — type_aliases.py

**Why:** The KV cache type `tuple[list[torch.Tensor], list[torch.Tensor]]` is repeated
across attention.py and decoder_layer.py, and model.py will need `list[KVCache]`. Without
named aliases this becomes unreadable. Centralising them is required before Unit 6.

**Aliases:**
- `KVCache = tuple[list[Tensor], list[Tensor]]` — single-layer cache
- `ModelKVCache = list[KVCache]` — full model cache (one entry per decoder layer)

**Testing:** Type aliases have no runtime behaviour. Verification is that all existing
tests continue to pass after the refactor.

---

### Unit 5 — decoder_layer.py

**What:** `DecoderLayer` — a single transformer block: pre-norm attention followed by pre-norm
MLP, with residual connections.

**Why pre-norm:** Normalising the sublayer *input* (not output) keeps the residual stream
unnormalised. Gradients flow more cleanly through unnormalised residuals at depth, and each sublayer
operates on a stable, normalised view of the signal. This is why large language models trained at
scale use pre-norm.

**Structure:**
```
normed = RMSNorm(x)
h      = x + Attention(normed, ...)
normed = RMSNorm(h)
out    = h + MLP(normed)
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

### Unit 6 — model.py (Llama3Model)

**What:** `Llama3Model` — the pure transformer backbone. No LM head, no loss, no HF generation
machinery. A clean, testable unit that can be verified before the HF wrapper is added.

**Why separate from Llama3ForCausalLM:** The backbone is independently verifiable. Mixing in the
HF contract, weight tying, and loss computation before the backbone is confirmed correct makes
failures harder to diagnose. Verify the foundation first.

**Structure:**
- Stack of `num_hidden_layers` `DecoderLayer` instances
- Final `torch.nn.RMSNorm` — the stack output is normalised before any projection
- No token embedding — the backbone is modality-agnostic; it accepts pre-embedded hidden states.
  Token embedding lives on `Llama3ForCausalLM`. This is the correct HF convention.

**forward() input:** `inputs_embeds: torch.Tensor` of shape `(batch, seq_len, hidden_size)` —
not token IDs. The caller is responsible for embedding tokens before calling the backbone.

**Returns a dict** with:
- `"last_hidden_state"`: output of the final decoder layer, shape `(batch, seq, hidden_size)`
- `"past_key_values"`: `ModelKVCache | None`
- `"hidden_states"`: tuple of per-layer outputs if `config.output_hidden_states` is True, else None

**HF contract (minimal for this unit):**
- Inherits `PreTrainedModel`
- `config_class = Llama3Config`
- `base_model_prefix = "model"`
- `post_init()` called at end of `__init__`

**Invariants that must hold (tested):**
- Output `"last_hidden_state"` shape: `(batch, seq, hidden_size)`
- KV cache: hidden state at position t with cache matches full forward at that position
- `output_hidden_states=True` returns one tensor per layer plus the inputs_embeds
- `output_hidden_states=False` returns None for hidden_states

---

### Unit 7 — huggingface.py (Llama3ForCausalLM)

**What:** `Llama3ForCausalLM` — HF wrapper around `Llama3Model`. Adds the LM head, weight tying,
loss computation, and the full HF AutoClass contract.

**Why separate from model.py:** The backbone (model.py) and the HF contract wrapper are distinct
responsibilities. The backbone transforms tokens to representations; the wrapper owns vocabulary
projection, loss, generation, and the AutoClass/save/load contract. One file per responsibility.

**Structure:**
- Token embedding: `nn.Embedding(vocab_size, hidden_size)` — lives here, not on the backbone
- Contains `Llama3Model` as `self.model`
- LM head: `nn.Linear(hidden_size, vocab_size, bias=False)`
- `tie_word_embeddings`: if True, `lm_head.weight` is directly assigned to `embed_tokens.weight`
  after `post_init()`. Both are `(vocab_size, hidden_size)` — same shape, no transpose needed.
  No `_tied_weights_keys` — not required; direct assignment is sufficient.

**forward():**
- Embeds `input_ids` via `self.embed_tokens`, passes `inputs_embeds` to `self.model`
- Projects last hidden state to logits via `self.lm_head`
- Computes cross-entropy loss if labels provided (labels shifted by one — each token predicts next)
- Returns a plain **dict** (not `CausalLMOutputWithPast`) with keys: `"logits"`, `"loss"`,
  `"past_key_values"`, `"hidden_states"`.

**HF contract:**
- Inherits `PreTrainedModel` and `GenerationMixin`
- `config_class = Llama3Config`
- `base_model_prefix = "model"`
- `_no_split_modules = ["DecoderLayer"]`
- `supports_gradient_checkpointing = True`
- `post_init()` called at end of `__init__`
- `_init_weights` overridden to no-op (Blocker B) — PyTorch constructor defaults stand

**Invariants that must hold (tested):**
- Output `"logits"` shape: `(batch, seq, vocab_size)`
- Loss computed correctly when labels provided (cross-entropy, next-token prediction)
- KV cache: generation with cache produces identical logits to full forward at current position
- `save_pretrained` / `from_pretrained` round-trip: all weights identical
- `AutoModelForCausalLM.from_config(config)` instantiates without error

---

### Blocker A — embed_tokens on Llama3ForCausalLM

**What:** Move `embed_tokens` from `Llama3Model` to `Llama3ForCausalLM`. `Llama3Model` becomes
a pure transformer stack that accepts hidden states `(batch, seq_len, hidden_size)` rather than
token IDs. This is the correct HF convention: the backbone is modality-agnostic; the token
interface lives on the task wrapper.

**Consequence for weight tying:** With both `embed_tokens` and `lm_head` on `Llama3ForCausalLM`,
tying is a direct assignment `self.lm_head.weight = self.embed_tokens.weight` in `__init__`,
after `post_init()`. No `_tied_weights_keys` machinery required. Both weights are
`(vocab_size, hidden_size)` — same shape, no transpose needed.

**Files affected:** `model.py`, `huggingface.py`, `test_model.py`, `test_huggingface.py`.

---

### Blocker B — override _init_weights to no-op

**What:** Override `_init_weights` on both `Llama3Model` and `Llama3ForCausalLM` to a no-op.
HF's default `_init_weights` silently reinitialises all Linear and Embedding weights with
`normal(0, 0.02)`, replacing PyTorch's constructor defaults. This is invisible and wrong for
our use case — we want PyTorch's own initialisations to stand.

**Safety:** PyTorch's module constructors call `reset_parameters()` unconditionally at
construction time, before HF's `_init_weights` runs. Overriding `_init_weights` to a no-op
does not suppress that — it only prevents HF's second pass from overwriting it. New heads
added later still get PyTorch's defaults from their constructors.

**Files affected:** `model.py`, `huggingface.py`.

---

### Blocker — Restructure src/llama3/ into src/llama3/model/ with relative imports

**What:** Move all model source files from `src/llama3/` into `src/llama3/model/`. Move all test
files from `tests/llama3/` into `tests/llama3/model/` to preserve the mirror invariant. Convert all
intra-package imports to relative imports (`from .configuration import Llama3Config` etc.).

**Why relative imports are required:** HuggingFace's `trust_remote_code` mechanism downloads the
contents of the Hub repository root into a local cache directory and adds that directory to
`sys.path`. In that context there is no `src` or `llama3` package — absolute imports break. Relative
imports work because `__init__.py` is present in `model/` and Python treats the cache directory as
a package. This is a prerequisite for the Hub distribution to function correctly.

**Why src/llama3/model/ is the Hub distribution unit:** `upload_folder` uploads the contents of a
local directory directly to the Hub repository root — no copying, no file manifest to maintain. The
folder's contents are exactly what researchers receive. This eliminates the possibility of drift
between the local source and what is on the Hub.

**Invariants that must hold:**
- All existing tests pass after the move with no changes to test logic — only import paths change
- `from .X import Y` style imports work correctly within the package
- The test directory structure mirrors the src directory structure exactly

**Files affected:** All files currently in `src/llama3/`, all files in `tests/llama3/`.

---

### Blocker — tokenizer.py: prepare GPT-NeoX tokenizer for Hub distribution

**What:** New file `src/llama3/tokenizer.py`. Single responsibility: ensure the GPT-NeoX tokenizer
files are present and correct in `src/llama3/model/` so that `upload_folder` includes them and
`AutoTokenizer.from_pretrained` succeeds after upload.

**Why GPT-NeoX (`EleutherAI/gpt-neox-20b`):** Byte-level BPE — no UNK tokens, all Unicode handled
without fallback. 50,277-token vocabulary: large enough for modern use without the embedding
parameter cost of 128K+ vocabularies at small hidden sizes. Apache 2.0 license, not gated. Trained
on The Pile, giving broader corpus diversity than alternatives at this vocabulary scale.

**Why separate from upload_to_hub.py:** Tokenizer preparation is a distinct responsibility from Hub
upload orchestration. One file, one job.

**Known issue:** Loading `PreTrainedTokenizerFast` and calling `save_pretrained` can silently
downgrade `tokenizer_class` in `tokenizer_config.json` from `"PreTrainedTokenizerFast"` to a slow
variant. This must be corrected before the files are in place, or `AutoTokenizer` will load the
wrong class.

**Invariants that must hold:**
- After this module runs, tokenizer files are present in `src/llama3/model/`
- `tokenizer_config.json` correctly identifies the fast tokenizer class
- Vocab size is 50,277
- Encode/decode round-trips correctly (text → ids → text is lossless for standard input)

**Testing:** `tests/llama3/model/test_tokenizer.py`. Tests requiring network access must be marked
as such. A bad test is worse than no test — if the correct network-free test strategy is unclear,
ask before writing.

---

### Unit 8 — upload_to_hub.py: publish architecture and tokenizer to HuggingFace Hub

**What:** Standalone script that publishes the model architecture and tokenizer to a HuggingFace Hub
repository so researchers can instantiate a freshly initialised model with no checkpoint.

**Why separate:** Operational concern, not architecture. No model code imports this. It runs once per
release.

**Canonical interface — upload_folder:** The contents of `src/llama3/model/` are uploaded directly
to the Hub repository root via `huggingface_hub.upload_folder`. This is the correct mechanism: it
produces a single atomic commit, requires no file manifest, and ensures the Hub repository root
contains exactly what is in the local folder — no more, no less. No other upload mechanism is used
for the model files.

**Canonical interface — authentication:** The script authenticates via a token stored by
`huggingface-cli login`. No token appears in code or in the repository. Researchers pulling from
the Hub need no credentials — all distributed files are public and the GPT-NeoX tokenizer is not
gated.

**Config section:** Target repository (`REPO_ID`) and any other upload-time settings are declared
in a clearly marked config section at the top of the script. Nothing is hardcoded deeper in the
script body.

**No weights:** The script never uploads model weights and must make it structurally impossible to
do so accidentally.

**Invariants that must hold (success conditions):**

When the upload script runs:

- `AutoConfig.from_pretrained("namespace/repo", trust_remote_code=True)` returns a valid
  `Llama3Config`
- `AutoModelForCausalLM.from_config(config, trust_remote_code=True)` instantiates `Llama3ForCausalLM`
  with fresh random weights and no errors
- `AutoTokenizer.from_pretrained("namespace/repo")` returns a working tokenizer
- A model card is visible on the Hub repository page
- The Hub repository contains no weight files
- Changes propogate into the huggingface directory.

**Testing:** Hub interaction requires a live connection and cannot be unit tested. Manual
verification against a test namespace is the appropriate strategy: run the script, then confirm all
five invariants above hold from a fresh environment with no local cache.

---

### Unit 9 — Documentation

**What:** Contributor-facing documentation covering development decisions, architectural rationale,
legal status, and anything a contributor needs to understand the codebase and its history.

**Why separate:** Distinct responsibility from the upload script and model card. This documentation
is for contributors navigating the codebase, not for researchers instantiating the model. It is not
pushed to the Hub. The full story is only known after development.

**Audiences and what they need:**

- User: knows how to set up the architecture on HuggingFace and instantiate a model. Covered by
  `documentation.md` (usage and upload steps already present).
- Developer: knows the `model/` WYSIWYG design, the relative-import constraint and why it exists,
  and where to find implementation history. Needs a developer section in `documentation.md` and
  `plan.md` co-located in `src/llama3/`.
- Legal: can verify the model is safe to use and fork. Needs `Legal.md`.

**Files produced:**

- `src/llama3/documentation.md` — expand existing file with a Developer Notes section: the
  `model/`-as-Hub-root WYSIWYG design, the relative-import requirement inside `model/` and why,
  and a pointer to `plan.md` for full implementation history.
- `src/llama3/Legal.md` — new file: MIT license statement, clean-room synthesis explanation
  (human coder never read Llama source; LLM synthesised the plan; implementing instances not shown
  raw Llama code), GPT-NeoX tokenizer under Apache 2.0.
- `src/llama3/plan.md` — this file, moved from repo root into `src/llama3/`, revised to its
  final state. The checklist is promoted near the top, preceded by an explanation of what this
  document is and how it was used: a human-supervised process record for LLM-assisted development,
  where each unit was planned, implemented, and verified before the next began, with all
  significant decisions surfaced for human review. The checklist is the visible evidence that this
  process was followed. This document is the primary trust artifact for the codebase — it is what
  allows a reviewer to answer "why should I trust LLM-produced code?" with something concrete.

**Success Invariant:**

- A user can follow `documentation.md` to upload and instantiate without reading anything else.
- A developer can read `documentation.md` and understand the structural quirks before touching code.
- Anyone can read `Legal.md` and understand why the codebase is safe to use and fork.
- A reviewer can open `plan.md`, immediately see what process was followed and that it was
  completed, and then drill into the full implementation history for any unit.

**Conditioning factors**

Development:

- Highly LLM-assisted; see `plan.md` for original prompt, history, and decisions.
- Intended as an easily usable Llama 3 baseline for research variants.
- Tokenizer choice: GPT-NeoX (`EleutherAI/gpt-neox-20b`) — byte-level BPE, 50,277 vocab,
  Apache 2.0, trained on The Pile.

Legal:

- Repo is under MIT license.
- Clean-room synthesis: the human coder has never read the Llama source. An LLM instance
  researched and produced the plan; separate instances implemented it without access to raw
  Llama code. No copyright violation is possible.
- GPT-NeoX tokenizer is Apache 2.0.

---

### Blocker — forward() return types: plain dict → ModelOutput

**Why:** `GenerationMixin.generate()` accesses `outputs.logits` and `outputs.past_key_values`
as attributes. Both `Llama3Model` and `Llama3ForCausalLM` currently return plain `dict`s,
which do not support attribute access and will crash immediately when `generate()` is called.
This must be resolved before Unit 10 tests can run.

**What:** Change `Llama3Model.forward()` to return `BaseModelOutputWithPast` and
`Llama3ForCausalLM.forward()` to return `CausalLMOutputWithPast`. Both are `ModelOutput`
subclasses that support dict-style key access (`output["logits"]`) in addition to attribute
access, so all existing tests continue to pass without modification.

**Files affected:** `src/llama3/model/model.py`, `src/llama3/model/huggingface.py`.

**Testing:** All existing tests in `test_model.py` and `test_huggingface.py` must continue
to pass after the change. No new tests — correctness of the types is verified by Unit 10.

---

### Blocker — _reorder_cache for beam search

**Why:** `GenerationMixin.generate()` calls `self._reorder_cache(past_key_values, beam_idx)`
when `num_beams > 1`. Without it, beam search raises `AttributeError`. Greedy decoding is
unaffected, but beam search is a standard use case that must not be silently broken.

**What:** Implement `_reorder_cache` on `Llama3ForCausalLM`. For each layer's cache, reindex
the batch dimension of every key and value tensor by `beam_idx`.

**Files affected:** `src/llama3/model/huggingface.py`.

**Testing:** Unit tests in `test_huggingface.py`: swap verifies correct reorder direction,
copy verifies beam collapse (beam_idx=[0,0] duplicates entry 0), structure check verifies
layer count and tensor shapes are preserved. Full pipeline verified by beam search test
in Unit 10.

---

### Blocker — Integration Tests

**Why:** Unit 10 tests the full user journey from the Hub. Before that journey can be
verified, we need confidence that the assembled model works locally — that `generate()` runs,
gradients flow, and local AutoClass instantiation succeeds. These are distinct
responsibilities: one verifies the assembled system in isolation; the other verifies the Hub
distribution path. Mixing them into a single unit blurs focus and makes failures harder to
attribute.

**What:** Local tests (no network) that verify the three use cases with a locally
constructed model. Any bugs discovered here (e.g. in generation or gradient flow) are
resolved as new blockers before Unit 10 begins.

**Test file:** `tests/llama3/test_end_to_end.py` — integration tests live here alongside
the end-to-end tests, separated into distinct classes.

**Tests:**
- **Generatable:** `model.generate(input_ids, max_new_tokens=5)` returns shape
  `(batch, input_len + 5)` and all token IDs are in `[0, vocab_size)`. Same input produces
  identical output across two calls (determinism).
- **Beam search:** `generate()` with `num_beams=2`, `use_cache=True` vs `use_cache=False`
  produce identical output. With `use_cache=False` correct by construction; with
  `use_cache=True` any `_reorder_cache` bug causes divergence.
- **Trainable:** `loss.backward()` runs without error and every parameter that should have
  a gradient has one.
- **HF-loadable:** `AutoModelForCausalLM.from_config(config)` instantiates correctly after
  local registration.

**Invariants that must hold:**
- `model.generate()` runs without error and produces valid token IDs
- Gradients flow to all trainable parameters
- Local AutoClass instantiation succeeds

---

### Unit 10 — End-to-End Tests

**What:** Tests that verify the full user journey starting from the Hub. The starting point
is always the Hub — never a locally constructed model. These replicate exactly what a
researcher does when they pull and use this library.

**Why separate from integration tests:** Integration tests confirm the assembled model
works. End-to-end tests confirm the Hub distribution path works: that the files on the Hub
are correct, that `trust_remote_code` loads the right classes, and that the resulting model
is usable. A passing integration suite does not imply a passing end-to-end suite.

**Three use cases:**
1. **HuggingFace-loadable** — config and model load from Hub via AutoClass
2. **Generatable** — Hub-loaded model produces valid output from `generate()`
3. **Trainable** — Hub-loaded model computes loss and gradients flow

**Test file:** `tests/llama3/test_end_to_end.py`

**Tests (`@pytest.mark.network`):**
- **Loadable:** `AutoConfig.from_pretrained(HUB_REPO, trust_remote_code=True)` and
  `AutoModelForCausalLM.from_config(config)` succeed. The resulting model is a
  `Llama3ForCausalLM` instance.
- **Generatable:** Load model from Hub, call `model.generate(input_ids, max_new_tokens=5)`,
  confirm output shape and all token IDs are in `[0, vocab_size)`.
- **Trainable:** Load model from Hub, run a forward pass with labels, call
  `loss.backward()`, confirm loss is finite and gradients exist on all trainable parameters.

**Invariants that must hold:**
- Hub load path produces a correct, usable `Llama3ForCausalLM` instance
- `model.generate()` produces valid token sequences
- Gradients flow to all trainable parameters

---

### Unit 11 — Audit

**What:** A first-principles review asking whether the codebase actually satisfies the
purpose stated in job.md. Not a verification that implementation matches plan — the plan
itself could be wrong. The auditor reads job.md, reasons independently about what must
be true for this system to work as described, and then examines the code with that
question in mind.

**The governing question:** A researcher wants to pull a configurable Llama 3-style
architecture from the Hub, instantiate it with fresh weights, and train or generate
from it. Does this codebase actually deliver that? Not in theory — in practice, with
the actual code as written.

**What the auditor must not do:** Use this plan as a checklist. The plan governed
construction and may share its blind spots. The auditor's value comes from reasoning
independently about what could be subtly wrong — mismatches between stated intent and
actual behaviour, assumptions that are never validated, contracts that are partially
satisfied, things that work in the tests but would fail in a researcher's hands.

**Output:** A written findings report. Each finding is classified as:
- **Defect** — a violation of job.md's requirements; must be resolved before the audit
  is complete
- **Observation** — a deviation from intent or best practice that does not strictly
  violate a requirement; surfaced for human review
- **Clean** — no issues found in this area

The audit is complete when all defects are resolved and the user has reviewed all
observations.

---

### Blocker — DynamicCache compatibility

**Why:** `GenerationMixin.generate()` passes a `DynamicCache` object as `past_key_values`
on the first forward call. Our `Llama3Model.forward()` indexes `past_key_values[i]` by
layer, which raises `TypeError: 'DynamicCache' object is not subscriptable`. This blocks
all generate()-dependent integration tests.

**Root cause and design decision:** `DynamicCache` is not subscriptable — our model's
`past_key_values[i]` indexing fails. The fix adopts `DynamicCache` natively.
`cache.update(new_k, new_v, layer_idx)` is the only documented interface: it stores the
new K/V and returns `(full_k, full_v)` — the full accumulated history for that layer.
Because the full K/V is needed for attention and is only available after projections,
`update()` must be called inside attention.py.

**Separation of concerns:** `huggingface.py` decides which Cache type to instantiate.
The model accepts any Cache object and calls `update()` on it — no knowledge of the
specific subclass. Future research swaps the cache type in huggingface.py only.

**New interface — attention.py:**
`GroupedQueryAttention.forward(hidden_states, position_ids, cache=None, layer_idx=0)`
- `cache`: any `Cache` subclass, or None when `use_cache=False`.
- `layer_idx`: identifies which cache slot to read/write via `cache.update()`.
- After computing projections: if cache is provided,
  `full_k, full_v = cache.update(k, v, layer_idx)`; else `full_k, full_v = k, v`.
- `is_causal`: `cache is None or cache.get_seq_length(layer_idx) == 0` — True on
  prefill (no history yet), False on cached decode steps.
- Returns `output` only. Cache is updated in place; K/V need not be returned.

**New interface — decoder_layer.py:**
`DecoderLayer.forward(hidden_states, position_ids, cache=None, layer_idx=0)`
- Threads `cache` and `layer_idx` into attention. Returns `hidden_states` only.

**New interface — model.py:**
`Llama3Model.forward(inputs_embeds, ..., past_key_values=None)`
- `past_key_values`: a `Cache` object or None. Threaded through to every layer.
- `position_ids`: when None and cache is provided, computed from
  `past_key_values.get_seq_length()`. This correctly offsets RoPE positions during
  cached generation.
- Iterates layers passing `cache=past_key_values, layer_idx=i`.
- `use_cache=True`: returns the same cache object as `past_key_values` — updated in
  place by each layer's attention call.
- `use_cache=False`: passes `None` as cache, returns `None` for `past_key_values`.

**New interface — huggingface.py:**
- `use_cache=True` and `past_key_values is None`: creates
  `DynamicCache(config=self.config)` before calling the model.
- `use_cache=True` and cache already provided: passes it through unchanged.
- `use_cache=False`: passes None.
- `_reorder_cache`: calls `past_key_values.reorder_cache(beam_idx)` and returns the
  cache object. DynamicCache handles beam reordering internally.

**type_aliases.py:** Delete. `KVCache` and `ModelKVCache` described the chunk-list
format, which no longer exists anywhere in the package.

**Files affected:**
- `src/llama3/model/attention.py`
- `src/llama3/model/decoder_layer.py`
- `src/llama3/model/model.py`
- `src/llama3/model/huggingface.py`
- `src/llama3/model/type_aliases.py` — deleted
- `tests/llama3/model/test_attention.py`
- `tests/llama3/model/test_decoder_layer.py`
- `tests/llama3/model/test_model.py`
- `tests/llama3/model/test_huggingface.py`

**Invariants that must hold (tested):**
- All existing invariants from Units 4–7 continue to hold under the new interface
- KV cache correctness: output at position t with cache equals output from a full forward
  pass — verified in `test_attention.py`, `test_model.py`, and `test_huggingface.py`
- `model.generate()` runs without error and returns valid token IDs
- `_reorder_cache` correctly delegates to `DynamicCache.reorder_cache()`

---

### Blocker — model.py pure torch refactor

**Why:** `Llama3Model` should have no HF lifecycle concerns. It is a pure transformer
stack — it takes tensors and returns tensors. Cache creation is a generation contract
decision; `use_cache` is a generation flag; `BaseModelOutputWithPast` is a HF container
type; `PreTrainedModel` brings save/load and `_init_weights` lifecycle machinery. None
of these belong on the backbone. All generation and HF contract concerns belong on
`Llama3ForCausalLM`. This was the original design intent, missed during earlier units.

**`model.py` changes:**
- Base class: `PreTrainedModel` → `nn.Module`
- Remove `config_class`, `base_model_prefix` class attributes
- Remove `_init_weights` override and `post_init()` call
- Remove `use_cache` parameter entirely — no flag, no DynamicCache import or creation
- Forward signature: `forward(inputs_embeds, position_ids=None, past_key_values=None, output_hidden_states=None)`
- If `past_key_values` is not None, pass it to layers and return it (updated in place).
  If None, pass None to layers and return None. No flags, no decisions.
- Return type: plain `dict` with keys `"last_hidden_state"`, `"past_key_values"`,
  `"hidden_states"`. `Llama3Model` has no dependency on HF output types.
- Remove `from transformers import PreTrainedModel, BaseModelOutputWithPast`
- Keep `from transformers.cache_utils import Cache` for the type annotation only.

**`huggingface.py` changes:**
- `use_cache` and `output_hidden_states` config resolution both live here. Before
  calling the backbone:
  - `use_cache = use_cache if use_cache is not None else self.config.use_cache`
  - `output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states`
  - `use_cache=True` and `past_key_values is None` → create `DynamicCache()`
  - `use_cache=False` → set `past_key_values = None` (ignore any provided cache)
- Access backbone output via dict keys: `backbone_out["last_hidden_state"]`,
  `backbone_out["past_key_values"]`, `backbone_out["hidden_states"]`.
- Both config fields remain live: config sets the default, per-call argument overrides it.

**`test_model.py` changes:**
- Remove `test_returns_base_model_output_with_past` — the return type is now dict.
- Remove `test_init_weights_is_noop` — `_init_weights` is gone with `PreTrainedModel`.
- Remove `test_attribute_access_works` — plain dicts don't support attribute access;
  replace with `test_dict_keys_present` verifying the expected keys are in the output.
- `test_use_cache_true_returns_one_entry_per_layer` → renamed and rewritten: pass a
  `DynamicCache()` explicitly as `past_key_values`; verify it comes back populated with
  `num_hidden_layers` entries.
- `test_use_cache_false_returns_none` → rewritten: pass no cache; verify
  `out["past_key_values"]` is None.
- `test_cached_generation_matches_full_forward` → updated to pass DynamicCache explicitly
  and use dict access throughout.
- `test_output_hidden_states_*` tests → updated to use dict access.
- `test_config_output_hidden_states_respected` → moved to `test_huggingface.py`; the
  config resolution now lives in `huggingface.py`, not `model.py`.

**Invariants that must hold (tested):**
- `Llama3Model` has no import from `transformers` except `Cache` for type hints
- Return value is a plain dict with keys `"last_hidden_state"`, `"past_key_values"`,
  `"hidden_states"`
- With cache provided: output at position t matches full forward — KV cache correctness
  unchanged
- With no cache provided: `past_key_values` is None in output
- All existing integration tests continue to pass (the change is internal to the backbone;
  `huggingface.py` owns all HF contract concerns)

**Files affected:**
- `src/llama3/model/model.py`
- `src/llama3/model/huggingface.py`
- `tests/llama3/model/test_model.py`

---

## Open Decisions

The following decisions were made during planning and are flagged for confirmation. They will be
revisited at the start of the relevant unit.

| # | Decision | Proposal | Status |
|---|----------|----------|--------|
| 1 | `model_type` | `"llama3_baseline"` | Confirmed by user |
| 2 | Attention kernel | `torch.nn.functional.scaled_dot_product_attention` | Confirmed by user |
| 3 | KV cache format | Originally chunk-list per layer; superseded by DynamicCache blocker — cache now lives entirely in `huggingface.py`, model receives plain concatenated tensors | Revised: DynamicCache compatibility blocker |
| 4 | YaRN scaling | Handled natively by HF's `ROPE_INIT_FUNCTIONS` — no placeholder needed | Resolved: all scaling types supported via HF |
| 5 | `intermediate_size` | Direct config parameter | Confirmed by user |
| 6 | Tokenizer | GPT-NeoX (`EleutherAI/gpt-neox-20b`), 50,277 vocab, Apache 2.0 | Confirmed by user |
| 7 | Hub upload mechanism | `upload_folder` from `src/llama3/model/` directly to Hub root — no copying, no manifest | Confirmed by user |
| 8 | Hub authentication | `huggingface-cli login` — token stored locally, nothing in code | Confirmed by user |
| 9 | `_tied_weights_keys` | Deliberately absent — direct assignment is sufficient | Confirmed by user |

---

