**# Implementation Plan: advanced-transformers-lib — SHRAM

## What This Document Is

This is the planning and process record for the implementation of the SHRAM (Sparse Hybrid Token
Routed Attention Mixture) architecture. It serves as both an active development tool and a trust
artifact. Every unit is planned here before code is written, every decision is recorded for human
review, and every unit is verified here before the next begins.

The process it records:

- Before writing any code for a unit, the plan for that unit was written and reviewed.
- Each unit was implemented, tested, and verified before the next began.
- When a blocker arose, it was pushed onto a stack, resolved as its own verified unit, and then
  the interrupted unit resumed. No blocker was hacked around inline.
- All significant decisions were surfaced for human review. The human author approved or corrected
  each one. Autonomous decisions were reported, not hidden.

This document does not define correctness — job.md does. This document records how correctness
is being achieved, one verified unit at a time.

---

## Completion Record

- [X] Preliminary — copy llama3 to shram, rename all identifiers, verify baseline tests pass
- [X] Unit 1 — audit the copy against SHRAM requirements; produce ordered change list
- [X] Unit 2 — independent verification of Unit 1 change list against the code
- [X] Unit 3 — ShramConfig: add all SHRAM-specific architectural parameters
- [X] Unit 4 — Router and load balancing: token-choice routing + DeepSeek biasing
- [X] Unit 5.A — Rope architecture design: how h_l and BEA construct independent RoPE instances
- [X] Unit 5.B — ShramConfig: replace HF rope infrastructure with explicit rope parameters
- [X] Unit 5.C — RotaryEmbedding: rewrite with explicit constructor and paper math
- [X] Unit 5.C.1 — RotaryEmbedding: revision pass (maintainability blocker; see unit)
- [X] Unit 6.0 — Cache architecture: situation statement, design decisions, folder refactor
- [X] Unit 6.A — MoSRAHCache and SlowMoSRAHCache: update/get_heads_lengths interface with oracle validation
- [X] Unit 6.B — ShramLayerCache: per-layer composite cache owning both sub-caches
- [X] Unit 6.C — ShramCache: top-level HF Cache owning all layer caches
- [ ] Unit 7 — Local sliding-window attention module (h_l)
- [ ] Unit 8 — Expert packing and unpacking: permutation machinery, padding, masks
- [ ] Unit 9 — Bottlenecked Ensemble Attention (BEA): per-head attention on packed tensors
- [ ] Unit 10.A — MoSRAH position computation: P tensor for both modes and all cache states
- [ ] Unit 10.B — MoSRAH sparse path: routing → packing → position computation → BEA → unpacking → weighted reduction
- [ ] Unit 11 — SHRAM hybrid layer: assemble H(x) = h_l (Unit 7) + h_s (Unit 10.B)
- [ ] Unit 12 — DecoderLayer: replace attention sublayer, propagate load_balance_loss
- [ ] Unit 13 — ShramModel: aggregate load_balance_loss in output
- [ ] Unit 14 — ShramForCausalLM: expose load_balance_loss; KV cache resolution
- [ ] Unit 15 — upload_to_hub
- [ ] Unit 16 — documentation
- [ ] Unit 17 — end-to-end tests
- [ ] Unit 18 — final audit

---

## Status

**Current state:** Units 6.A–6.C complete (127/127 tests passing). Unit 7 next. Network tests deselected (Hub repo not yet created).

---

## Governing Principles

This plan records not just *what* to build but *why* each decision was made, so future sessions
can make good changes without reconstructing lost context. Every design decision includes its
rationale. Every component includes the invariants its tests must enforce. If an invariant changes,
the tests must change to match before the unit can be considered verified.

The governing objective is correctness. Refer to job.md for the full statement of what that means
and why productivity is not a substitute. Work that cannot be verified against job.md's invariants
is not progress.

The architecture is a **synthesis**, not a transcription. The paper conditions the implementation
but does not dictate its structure, style, or organisation. The Llama 3 baseline in this repository
sets the code quality bar. Where the paper leaves choices open or is silent, a decision is made,
recorded here, and flagged for user review. Where the correct answer is unknown, work stops and
the user is asked.

---

## Process Rules

These are non-negotiable. Deviating without explicit user approval is not acceptable.

**One unit at a time.** Never begin a new unit while the current one is unverified. A unit is not
complete until its tests pass and accurately describe the intended behaviour.

**Blocker stack.** When completing a unit requires work outside its scope, that work is a new unit
pushed onto a stack. Complete and verify the blocker, then return. Only make the changes needed
for compatibility — do not expand scope.

**Surface decisions.** Autonomous decisions are permitted but must be reported to the user for
review. Uncertain decisions must be escalated before proceeding. Do not resolve ambiguity silently.

**Keep this plan current.** Update the status section and unit checklist continuously. The plan
must reflect actual state at all times so work can be resumed after a session break without loss.

**Plan first within each unit.** At the start of each unit, state the invariants the unit must
satisfy before considering any implementation. Implementation follows from invariants. A unit
whose implementation is planned before its invariants are stated is not planned — it is guessed.

**Close the testing gap.** When a defect is found — in audit or anywhere else — the resolution is
not complete until two questions are answered: (1) what invariant did the existing tests fail to
enforce, and (2) what test change closes that gap? The fix and the test correction are a single
unit of work.

**Surface paper↔implementation conflicts.** Where the paper is silent, incomplete, or leaves a
value undetermined, that is a decision point — not a license to fill the gap. Surface it, resolve
it with the user, and record the resolution here before proceeding.

---

## Code Quality Standards

These apply without exception to every file.

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

---

## Testing Philosophy

A codebase that works but cannot be verified has no value for research. Only a verified
implementation can be used to draw scientific conclusions. **Verified-but-imperfect is more
valuable than working-but-unverified** — only the former can be trusted as a research baseline.

Tests are first-class artifacts. They are written alongside the implementation, not appended
afterward. A component without passing tests is not complete regardless of how correct it appears.

**Rules:**
- Each src file has a corresponding test file mirroring the structure under `tests/shram/`
- Unit tests verify each component in isolation
- Integration tests verify that combinations of components work together — they do not replicate
  the detail of unit tests
- Placeholders may fail, but must raise `NotImplementedError`
- When a unit is modified as a blocker, its tests must be updated to reflect the new correct
  behaviour before the unit is re-verified. Passing tests that no longer reflect intent are false
  confidence.
- **A bad test is worse than no test.** When uncertain how to correctly test a component, ask the
  user before writing the test.

---

## Units of Work

---

### Preliminary — Copy and Rename

**What:** `src/shram/` and `tests/shram/` have been copied from `src/llama3/` and `tests/llama3/`.
All class names, module references, string identifiers, and docstring references must be renamed
from `Llama3`/`llama3` to `Shram`/`shram`. `model_type` updated to `"shram"`. The existing
llama3 plan.md has been replaced with this document. Baseline tests must pass after rename before
this unit is marked complete.

**Invariants this unit must satisfy:**
- All identifiers referencing Llama3 or llama3 have been renamed to Shram or shram throughout
  `src/shram/` and `tests/shram/`.
- `model_type` is `"shram"`.
- All tests in `tests/shram/` pass on the unmodified (renamed) copy.
- `src/llama3/` and `tests/llama3/` are untouched.
- No functional changes have been made — this unit is rename only.

---

### Unit 1 — Audit the Copy; Produce the Change List

**What:** Read the paper against every file in `src/shram/`. For each file, determine: does it
transfer unchanged, does it need modification, or does it need replacement? For each file that
needs modification or replacement, describe what must change and why, grounded in the paper.
Identify all open decisions that must be resolved before that work can begin.

The output of this unit is a fully populated change list appended to this plan — one entry per
affected file, with rationale and open decisions called out. Units 2..N in the completion record
are filled in at this point. No code is written in this unit.

**Why it exists:** The scope of changes is not yet known. Committing to a unit breakdown before
understanding the problem produces a plan that will be wrong. This unit produces the evidence
the plan needs before implementation begins.

**Philosophical invariants** (define what correct output means — guide edge case decisions):
1. Every file in `src/shram/` has been assessed against the paper. No file is skipped or assumed.
2. Minimality — changes are proposed only where the paper requires them. A file that satisfies
   SHRAM requirements as-is is correctly assessed as "transfers unchanged."
3. Every proposed change has a rationale traceable to a specific paper requirement — a named
   section, claim, or architectural invariant. Not "this seems necessary."
4. Every open decision is explicitly named and surfaced. Where the paper is silent, ambiguous,
   or undetermined, that gap is a decision point — not filled silently.
5. The plan produced is executable by a future instance without reconstructing this session's
   reasoning. Every claim is specific and verifiable.
6. The change list is complete — necessary and sufficient. No required change is missing.
7. No code has been written or modified.

**Operational invariants** (define what correct process looks like — things to do in known cases):
1. Paper is read in main context. Synthesis cannot be delegated.
2. Specialist Explore agents are used per logical area. Code understanding is not built from
   memory or assumption.
3. Specialists are probed iteratively via SendMessage until answers are consistent and specific.
   The first answer is not trusted.
4. The change list is built incrementally as each area is settled — not constructed all at once
   from potentially stale memory.
5. An audit unit is written into the plan with specific verifiable claims. A future instance
   executes it independently.
6. The audit unit contains specific verifiable claims, not vague directives.
7. Memories are updated consistently throughout this unit so future instances can resume
   without loss of context.

**Unit 1 Findings — Variable Naming**

Paper uses single-letter math symbols. Code must use descriptive names. The mapping below
is authoritative — implementations use these names, not the paper's symbols.

| Paper symbol | Code name | Purpose |
|---|---|---|
| Π | `expert_order_idx` | Stable-sort permutation: converts token-choice to expert-choice order |
| Π⁻¹ | `token_order_idx` | Inverse permutation: restores expert-choice to token-choice order |
| J | `packed_positions` | Original sequence positions of packed tokens (B, L, T) |
| M | `active_mask` | Boolean mask: True for real tokens, False for padding (B, L, T) |
| I | `selected_heads` | TopK head indices per token (B, N, K) |
| P | `routing_probs` | Unbiased renormalized routing probabilities (B, N, K) |
| R | `routing_scores` | Unbiased routing activations from Softmax(xW_r) |
| R̂ | `biased_routing_scores` | Biased activations Softmax(xW_r + expert_bias) used for TopK |
| b | `expert_bias` | Learned per-head bias for load balancing |
| u | `mosrah_head_dim` | BEA bottleneck width = hidden_size // num_selected_heads |
| x' | `packed_hidden` | Packed expert-choice hidden states (B, L, T, d) |
| ỹ | `unpacked_output` | Unpacked token-choice responses (B, N, K, d) |

**Unit 1 Findings — Change List**

These are the output of Unit 1. Every claim is traceable to a paper section. Unit 2 (audit)
verifies these claims before implementation begins.

*Files that transfer unchanged:*

- `tokenizer.py` — GPT-NeoX tokenizer with 50,277 vocab is explicitly specified (paper §4.3
  Tokenizer). Implementation matches. No architectural dependency on the attention mechanism.

- `model/mlp.py` — SwiGLU FFN is explicitly specified (paper §4.3). Formula matches:
  down_proj(silu(gate_proj(x)) ⊙ up_proj(x)). No changes required.

- `__init__.py`, `model/__init__.py` — No architectural logic. Will need mechanical export
  updates as new files are added; that work is part of whichever unit introduces the new file.

*Files that need modification:*

- `model/configuration.py` — SHRAM requires parameters absent from the current config: total
  MoSRAH heads L, heads selected per token K, local attention window size, local attention head
  parameters, RoPE mode (main-sequence vs semantic-sequence). Existing GQA parameters may need
  repurposing or removal depending on OD-3. Paper refs: §4.3, Appendix A.Local Attention,
  Appendix B.

- `model/rope.py` — BEA requires RoPE on position tensor `packed_positions` of shape (B, L, T)
  with head_dim `mosrah_head_dim` = d/K. Current interface is (B, S) with config.head_dim —
  incompatible. Resolution: extend rope.py to accept a position tensor of arbitrary leading
  shape via direct tensor indexing (see OD-4). Also verify YaRN implementation independently.
  Likely surfaces as a blocker during Unit 6. Paper refs: Appendix B.RoPE Mechanics.

- `model/decoder_layer.py` — Currently returns a single tensor. SHRAM hybrid layer emits
  load_balance_loss alongside the output. DecoderLayer must propagate it. Pre-norm + residual
  structure transfers unchanged (paper Appendix A confirms it).

- `model/model.py` — Output dict has no load_balance_loss key. Job.md Architecture §:
  "The load-balance loss must appear in the model's forward output." ShramModel must
  aggregate load_balance_loss across decoder layers and include it in output.

- `model/huggingface.py` — load_balance_loss must be accessible from the forward output so
  the training loop can weight and apply it. KV cache: DynamicCache virtual layer indexing —
  each SHRAM layer computes its own indices as `layer_idx * num_mosrah_heads + head_idx`;
  see OD-2 resolution. Paper refs: §4.3 Pretraining (auxiliary loss weight 1e-2), job.md §.

*Files that need replacement:*

- `model/attention.py` — GroupedQueryAttention is not the SHRAM attention mechanism. SHRAM
  requires H(x) = h_l(x) + h_s(x): local sliding-window attention plus MoSRAH. The current
  file has no sliding window, no routing, no packing — it is standard GQA which SHRAM does
  not use. Replacement introduces these major new responsibilities: token-choice routing and
  load balancing, expert packing/unpacking, Bottlenecked Ensemble Attention, MoSRAH sparse
  path assembly, SHRAM hybrid layer. File granularity for these responsibilities is decided at
  unit planning time for Units 3–7. Paper refs: §3 Design, Appendix A throughout.

**Open Decisions**

Resolutions recorded here. Unresolved items must be resolved before the indicated unit begins.

- **OD-1: Sliding window kernel** — Local path h_l requires a kernel that performs local
  attention natively: the window is enforced by the kernel itself, not by a boolean attn_mask
  constructed outside it. The invariant is kernel-native local attention; the specific library
  (flash_attn, flex_attention, or other) is secondary and is determined at Unit 5 planning
  time. The unit planner evaluates available options against this invariant and records the
  choice and rationale in this plan before implementation begins.
  Must resolve before Unit 5.
  Resolution: [UNRESOLVED — unit planner determines at Unit 5 planning]

- **OD-2: KV cache mechanism for MoSRAH** — Routing is dynamic: each decode step a token
  selects K of L heads and only those heads' caches are updated. MoSRAH's routing produces
  a ragged distribution of token counts across (batch, head) slots — different batch items
  may route different numbers of tokens to the same head in a single forward pass.
  DynamicCache cannot represent this: it stores one tensor per slot and concatenates along
  the sequence dimension, assuming a uniform token count across the batch. Forcing the
  ragged structure into DynamicCache would require external bookkeeping that bypasses the
  cache entirely, which is not acceptable.
  `MoSRAHCache` (Units 6.A.A and 6.A.B) therefore uses a custom buffer: `self.keys` and
  `self.values` each of shape `(B, L, T_max, u)` per layer, with a `(B, L)` integer count
  tensor `self.counts` tracking valid occupancy per (batch, head) slot. `T_max` doubles
  when any slot overflows. Data beyond the count is junk; consumers use
  `get_heads_lengths()` to obtain the counts and are responsible for masking.
  Must implement before Unit 11.
  Resolution: RESOLVED — custom buffer design; DynamicCache virtual-index approach
  superseded due to ragged-batch routing.

- **OD-3: GQA in local attention** — Local path uses standard MHA (not GQA). Paper says
  "standard causal sliding-window attention"; standard means MHA. num_key_value_heads is
  removed from ShramConfig; num_attention_heads is repurposed as local attention head count.
  Must resolve before Unit 3.
  Resolution: RESOLVED — MHA for local path, remove GQA params.

- **OD-4: RoPE interface for BEA** — rope.py is extended to accept a position tensor of
  arbitrary shape via direct tensor indexing: `cos = cos_cache[position_ids]`. This naturally
  handles both 2D (B, S) for existing usage and 3D (B, L, T) for BEA. No separate BEA-local
  RoPE. YaRN correctness is verified independently as part of this unit.
  Must implement before Unit 6.
  Resolution: RESOLVED — extend rope.py with arbitrary-shape position tensor support.

- **OD-5: RoPE mode** — Both main-sequence (`packed_positions` = original token positions)
  and semantic-sequence (`packed_positions` = local slot indices 0,1,2,...) modes must be
  implemented. A config flag selects between them. Experimentally correct mode is
  undetermined (paper §4 Hyperparameter Tuning) — both are required.
  Config parameter defined in Unit 3; implementation in Unit 6 (BEA).
  Resolution: RESOLVED — both modes, config flag.

---

### Unit 2 — Independent Verification of the Change List

**What:** A future instance verifies the Unit 1 change list claims against the actual code.
No implementation begins until this unit passes. Haiku agents are dispatched with specific
verifiable questions — one per claim that can be wrong.

**Invariants this unit must satisfy:**
- Every "transfers unchanged" claim is verified: the file contains no mechanism that SHRAM
  requires to differ from the current implementation.
- Every "needs modification" claim is verified: the named change is actually absent from the
  current code (i.e., the modification is genuinely needed).
- Every "needs replacement" claim is verified: the current mechanism is genuinely incompatible
  with SHRAM requirements.
- No required change is missing: the auditor finds no SHRAM architectural requirement that
  the change list fails to address.
- Every open decision is genuine: the named gap is actually unresolved in the paper and the
  code.
- If any claim is wrong, the discrepancy is surfaced and Unit 3 does not begin until resolved.

**Specific claims to verify (each dispatched as a targeted question to a haiku agent):**

1. mlp.py transfers unchanged — confirm SwiGLU formula is correct and no conflicting constants
2. tokenizer.py transfers unchanged — confirm no architectural dependency on attention
3. attention.py needs full replacement — confirm no sliding window, no routing, no packing
4. configuration.py needs modification — confirm no L, K, window_size, rope_mode currently
5. decoder_layer.py needs modification — confirm forward returns single tensor, no loss output
6. model.py needs modification — confirm output dict has no load_balance_loss key
7. huggingface.py needs modification — confirm no load_balance_loss in current forward output
8. rope.py needs modification — confirm position_ids is 2D only (B, S), not 3D

---

### Unit 3 — ShramConfig: SHRAM-Specific Parameters

**Invariants this unit must satisfy:**
- ShramConfig contains all parameters required by the SHRAM architecture that are not
  derivable from others: `num_mosrah_heads` (L), `num_selected_heads` (K), local attention
  window size, local attention head count and head_dim, `rope_mode` (main-sequence vs
  semantic-sequence).
- A single `head_dim` parameter (default 16 per paper §4.3) is shared by both the local
  sliding-window path and the MoSRAH/BEA path. It is explicitly specified — not derived
  from other parameters. There is no separate `mosrah_head_dim` config parameter.
- num_key_value_heads is removed (OD-3 resolved: local path uses MHA). num_attention_heads
  is repurposed as local attention head count.
- Every new parameter has validation that catches impossible configurations (e.g.,
  hidden_size must be divisible by num_selected_heads so that mosrah_head_dim is integral).
- Parameters that do not exist yet but may be needed later can be added as blockers when
  discovered during implementation — this config is not required to be exhaustive on first
  pass, only correct.
- Every parameter has a docstring tracing its architectural role to the paper.
- Tests cover: valid construction, every validation failure, all new parameters.

---

### Unit 4 — Router and Load Balancing

**Invariants this unit must satisfy:**
- The router produces `selected_heads` (B, N, K) and `routing_probs` (B, N, K) as specified
  in Appendix A.Routing.
- `selected_heads` is determined by TopK applied to `biased_routing_scores` =
  Softmax(xW_r + `expert_bias`). `routing_probs` is computed from `routing_scores` =
  Softmax(xW_r), gathered at `selected_heads` indices and renormalized. These two paths
  are not confused — `expert_bias` influences selection only, not probabilities.
- `expert_bias` is part of the router's learnable state and participates in the computation
  graph so that the custom gradient operator can write to it through the optimizer.
- Load balance loss = sum_l |f_l - 1/L| is emitted as a scalar from the router forward.
- The backward pass writes `L_grad * sign(f_l - 1/L)` to `expert_bias` via a custom autograd
  Function. Routing inputs (x) do not receive a gradient from this operator.
- File granularity (one file or two for routing vs load balancing) is decided at unit planning
  time, applying the one-responsibility-per-file policy.
- Tests verify: biased/unbiased scores are distinct, gradient flows to `expert_bias` not to
  x, load balance formula is correct, TopK correctness, custom backward correctness.

**Paper refs:** Appendix A.Routing, Appendix A.Load Balancing, Appendix A.Implementation
Concerns.

---

### Unit 5.A — Rope Architecture Design (complete)

**Decision:** h_l and BEA each construct and own an independent `RotaryEmbedding` instance.
No shared instance. Each path's rope behavior is fully determined by what it passes to its
own constructor — no config reading inside `RotaryEmbedding`.

**h_l constructs:** `RotaryEmbedding(mode="default", head_dim=config.head_dim, theta=config.local_rope_theta)`
Always standard RoPE. The returned `A_rope` is always 1.0 and may be discarded.

**BEA constructs:** `RotaryEmbedding(mode="yarn", head_dim=config.head_dim, theta=config.mosrah_rope_theta, initial_seq_length=config.training_sequence_length, dilation=config.scale, alpha=config.alpha, beta=config.beta)`
YaRN from paper §A.2. When `inference_sequence_length == training_sequence_length`, `s=1`
and YaRN reduces exactly to standard RoPE — this is the intended usage for researchers not
doing context extension, and must be documented.

**HF rope infrastructure:** Removed. ShramConfig no longer delegates rope initialisation to
HF's `RotaryEmbeddingConfigMixin`. All rope parameters are explicit config fields. No
`rope_scaling`, no `rope_parameters`, no `ROPE_INIT_FUNCTIONS`.

**Paper ref:** §A.2 RoPE Treatment, Appendix B.RoPE Mechanics.

---

### Unit 5.B — ShramConfig: Explicit Rope Parameters

**What:** Replace the HF-delegated rope infrastructure in `configuration.py` with explicit
research parameters. All previous rope-related config fields (`rope_theta`,
`max_position_embeddings`, `rope_scaling`, `yarn_alpha`, `yarn_beta`) are removed and
replaced with the fields decided in Unit 5.A.

**Invariants this unit must satisfy:**
- Config has exactly these rope-related fields: `local_rope_theta`, `mosrah_rope_theta`,
  `training_sequence_length`, `inference_sequence_length`, `alpha`, `beta`.
- A `scale` property returns `inference_sequence_length / training_sequence_length`. It is
  computed, not stored.
- No `rope_scaling`, `rope_theta`, `max_position_embeddings`, `yarn_alpha`, `yarn_beta`
  fields exist anywhere in the config.
- No HF rope mixin behaviour: `rope_parameters` is not produced and not expected by any
  downstream code.
- All fields survive a `to_dict` / `from_dict` serialisation roundtrip.
- Default values match the paper: `local_rope_theta=10000.0`, `mosrah_rope_theta=10000.0`,
  `alpha=1.0`, `beta=32.0` (paper §A.2 LLaMA-family recommendation).
  `training_sequence_length` and `inference_sequence_length` defaults are equal so that
  `scale=1` and the model is in standard-RoPE mode unless explicitly extended.
- Tests verify: all fields stored, `scale` property correct, roundtrip preservation,
  `scale=1` when lengths are equal.

**Paper ref:** §A.2 RoPE Treatment.

---

### Unit 5.C — RotaryEmbedding: Explicit Constructor and Paper Math

**What:** Rewrite `rope.py`. Remove all HF dependencies. Constructor accepts explicit
mode and parameters; each caller constructs with exactly what it needs.

**OD-4 resolution preserved:** `forward` accepts `position_ids` of arbitrary shape via
direct cache indexing — `cos = _cos_cached[position_ids]`. This handles both 2D `(B, N)`
for h_l and 3D `(B, L, T)` for BEA through the same code path.

**Invariants this unit must satisfy:**
- Constructor signature: `(mode, head_dim, theta, training_seq_len=None, inference_seq_len=None, alpha=None, beta=None, device=None)`.
- `mode="default"`: computes `dim_rotation_freqs = 1/theta^(2d/head_dim)`. Returns `A_rope=1.0`.
- `mode="yarn"`: computes YaRN-adjusted frequencies θ_d' per paper §A.2 equations exactly.
  Returns `A_rope = (0.1·ln(s)+1)²` where `s = inference_seq_len / training_seq_len`.
  When `s=1`, θ_d' = θ_d and `A_rope=1.0` — YaRN reduces to standard RoPE.
- Unsupported mode raises `NotImplementedError`.
- `yarn` mode requires `training_seq_len`, `inference_seq_len`, `alpha`, `beta` — raises
  `ValueError` if any are absent.
- `forward(q, k, position_ids)` returns `(q_rotated, k_rotated, A_rope)`. Works for
  `position_ids` of any integer tensor shape. Head dimensions in q/k are handled by
  inserting broadcast dimensions automatically.
- No `PretrainedConfig`, no `ROPE_INIT_FUNCTIONS`, no HF imports.
- Tests verify: default mode math (identity at position 0, relative position property),
  YaRN frequency formula matches paper equations, `s=1` produces identical output to
  default mode, `A_rope` values correct for both modes, arbitrary position_ids shape
  (2D and 3D), 3D gather correctness.

**Paper refs:** §A.2 RoPE Treatment, Appendix B.RoPE Mechanics.

---

### Unit 5.C.1 — RotaryEmbedding: Revision Pass (Maintainability Blocker)

**What:** The first-draft rope.py (Unit 5.C) passes all tests but has maintainability
problems identified in review. This unit corrects them before 5.C can be marked complete.
No new behaviour is introduced — this is a revision of the same module.

**Problems to fix:**
- Constructor signature uses `training_seq_len`/`inference_seq_len` — caller should pass
  `initial_seq_length` and `dilation` (the config's `scale` property) so arithmetic stays
  in the caller, not in RoPE.
- Validation guards are currently inline in `__init__`, breaking comprehension flow.
  Extract to individual private helpers — one per thing checked (e.g. `_validate_mode`,
  `_validate_yarn_params`).3 Called in sequence at the top of `__init__` before any
  computation; each is tiny and named so the constructor reads as a flat list of
  precondition checks followed by construction logic.
- Cache is a class-level dict on `RotaryEmbedding`, not a module-level variable.
- Cache tensors must be plain instance attributes, not registered buffers. Registered
  buffers are copied (not aliased) on `.to()`, silently breaking sharing. Only
  `rotation_freqs` remains a buffer for device mobility.
- `rotation_freqs` (was `dim_rotation_freqs`) has `head_dim/2` entries — one per dimension
  pair. Name must reflect this.
- Variable names throughout must be descriptive: `r` → `normalized_freqs`,
  `gamma` → `blend_weights`, `emb` → `angle_embedding`.
- `freq_key` constructed with explicit if/else, not tuple concatenation.
- Cache validity check in `forward` decomposed into named booleans, not a single compound
  condition.
- Broadcast-dimension loop: `while cos.ndim < q.ndim:` not `for _ in range(...)`.
- YaRN docstring must explain the three-regime big picture and cite the YaRN paper.

**Invariants this unit must satisfy:** All invariants from Unit 5.C hold unchanged.
All 26 tests must continue to pass. No new tests required unless a renamed interface
requires test updates.

---

### Unit 6 — Cache Architecture: Situation, Decisions, Folder Refactor

Overall, a competing set of circumstances shaped the implementation of the cache system

- Huggingface desired to use a Cache class and a per layer CacheLayerMixin to handle the concrete responsibilities. 
- BEA and the sliding window attention formulation desire individual caches for a custom cache approach to handle expert-choice caching and sliding window attention. All at one 'attention' slot in the standard transformer cache.
- The SHRAM layer exists to decode and dispatch as needed.

These together resulted in the following series of decisions.

- Top level cache will be a huggingface Cache object as standard.
- Each huggingface cache will be implemented as a series of ShramLayerCache using CacheLayerMixin. 
- Each ShramLayerCache will in turn contain a .sliding_window_cache (provided by huggingface) and a .mosrah_cache to handle the custom varieties of it.

This is correct as it most strongly lines up with the huggingface ecosystem while still responding to the necessities of the architecture. 

### 6.A MoSRAH cache

**Responsibility:** Define and verify the sparse cache subsystem for one MoSRAH decoder layer.

**Context of Correctness**

The MoSRAH system executes KV caching in expert-choice form, and its natural downstream
consumer, BEA, also works in expert-choice form. While other options were explored — most
notably a separate token-choice inference pathway — for the initial probe work it was decided
to keep this simple. As such, the cache simply asks for keys, values, and an active-token mask;
it inserts only active tokens into the underlying cache and returns the complete KV state with
an active-token mask for attention processes to use if desired. Additionally, the occupancy of
each individual head is critical information required for RoPE encoding and is exposed for
downstream usage.

**Invariants this unit must satisfy:**
- The unit defines the MoSRAH sparse cache for one decoder layer only. It does not own or
  represent sparse state for any other layer.
- The cache fulfills the HuggingFace per-layer cache role through `CacheLayerMixin`.
- The cache implements API `.update(keys, values, active_mask)`.
- `.update(keys, values, active_mask)` accepts expert-choice inputs, stores only active entries
  in causal order, and returns the full accumulated `(keys, values, active_mask)` across the
  cached sparse sequence for that layer.
- The returned `active_mask` correctly distinguishes valid cached positions from unwritten or
  junk positions.
- The cache implements `get_heads_lengths()` and returns correct per-head occupancies for the
  layer.
- All necessary HuggingFace cache methods have been evaluated and implemented to a locally sane
  standard.
- Any HuggingFace cache method, such as `get_seq_length()`, that does not make sense in a ragged
  packed sparse context instead raises `NotImplementedError()`.
- The cache supports repeated updates across multiple forward passes, with later updates
  extending prior sparse state rather than overwriting it.
- The cache implements reset and reorder behavior to a locally sane standard for HuggingFace
  cache usage.
- All mechanisms are implemented for very rapid usage.
- Tests and design are done in units which are verifiably correct as well.

**Tests**
- Verify `.update(keys, values, active_mask)` accepts expert-choice inputs and returns the full
  accumulated `(keys, values, active_mask)` in the expected format.
- Verify only active entries are inserted.
- Verify causal insertion order is preserved within each `(batch, head)` slot.
- Verify returned `active_mask` correctly identifies valid cached positions.
- Verify `get_heads_lengths()` returns correct per-head occupancies before updates, after
  updates, after repeated updates, and after reset.
- Verify repeated updates extend prior sparse state rather than overwriting it.
- Verify reset restores fresh-cache behavior.
- Verify reorder behavior permutes all sparse cached state consistently.
- Verify required HuggingFace per-layer cache behaviors are implemented sanely.
- Verify unsupported operations fail explicitly.
- If the final implementation is not obviously correct by inspection, certify it against a slow
  oracle implementation.

**Preliminary implementation strategy:**
- Verifiable correctness may make a slow, for-loop-based strategy that is then used as an oracle
  the preferred testing method.
- A large unit of memory initialized at construction across batches, heads, and sequences, and
  doubled in length any time storage runs low, is an excellent candidate for the underlying
  storage. A corresponding set of `(batch, head)` occupancy counters can then maintain the
  active-token masks and expose essential downstream information.
- It should be possible to use the following high-level formulation to quickly compute scatter
  indices under that design:
  1. Count the active sequence lengths and form an `arange` to that length.
  2. Add `get_heads_lengths()` to it. This broadcasts into the available sequences in order and
     gives the locations to insert into.
- If direct scatter over the provided storage is used, overwriting inactive or junk positions is verified to be fine. Masking will later ignore it, and any junk data inserted will just be overwritten in the next insertion cycle. 

### 6.B SHRAM layer cache

**Responsibility:** Define and verify the cache subsystem for one SHRAM decoder layer. This unit
owns, exposes, and transparently coordinates the two cache varieties required at a single SHRAM
attention slot: the sliding-window cache for the local path and the MoSRAH sparse cache for the
sparse path.

**Context of Correctness**

A SHRAM decoder layer contains two distinct cache-bearing attention pathways at one attention
slot. The local attention path wants a sliding-window cache, while the MoSRAH path wants the
custom expert-choice sparse cache defined in 6.A. At the same time, HuggingFace wants cache
responsibilities to exist at the per-layer level. For these reasons, the correct unit here is a
`ShramLayerCache` that owns both `.sliding_window_cache` and `.mosrah_cache`.

This unit should behave externally as one layer cache wherever a truthful composite behavior
exists, while still exposing the two internal cache systems directly for downstream usage by the
corresponding attention pathways. This is more correct than attempting to merge the two cache
systems behind a fake unified interface, because the two paths have different semantics and
different downstream consumers. The SHRAM layer cache therefore exists as the ownership,
dispatch, and transparent-composite boundary for one decoder layer.

**Invariants this unit must satisfy:**
- The unit defines the cache subsystem for one SHRAM decoder layer only. It does not own or
  represent cache state for any other layer.
- The cache fulfills the HuggingFace per-layer cache role through `CacheLayerMixin`.
- The cache owns exactly two sub-caches:
  - `.sliding_window_cache` for the local sliding-window path
  - `.mosrah_cache` for the MoSRAH sparse path
- The two sub-caches remain distinct and are not merged behind a misleading fake common cache
  interface using .update
- The unit exposes both sub-caches directly for downstream usage by the corresponding attention
  pathways.
- The unit transparently passes along reset and reorder commands to both sub-caches.
- The unit behaves externally as a single SHRAM layer cache wherever a truthful composite
  behavior exists, even though internally it owns multiple cache subsystems.
- Any HuggingFace per-layer cache method with a truthful composite meaning is implemented at the
  SHRAM layer cache boundary and dispatched appropriately to the owned sub-caches.
- Any HuggingFace cache method whose semantics do not truthfully apply at the composite SHRAM
  layer-cache boundary instead raises `NotImplementedError()`.
- All necessary HuggingFace per-layer cache methods have been evaluated and implemented to a
  locally sane standard.
- The unit is implemented for very rapid usage.
- Tests and design for this unit are themselves structured so the unit can be trusted as a
  verified black box by later cache and attention units.

**Tests**
- Verify the unit owns both `.sliding_window_cache` and `.mosrah_cache`.
- Verify the unit exposes the two sub-caches in a form directly usable by their downstream
  attention paths.
- Verify reset behavior clears both sub-caches through the SHRAM layer cache boundary.
- Verify reorder behavior permutes both sub-caches consistently through the SHRAM layer cache
  boundary.
- Verify any supported HuggingFace per-layer cache operation exposed at the SHRAM layer boundary
  dispatches correctly to the owned sub-caches.
- Verify unsupported composite operations fail explicitly rather than partially mutating only one
  sub-cache or returning misleading results.
- Verify required HuggingFace per-layer cache behaviors are implemented sanely.

**Preliminary implementation strategy:**
- If no contradictory ground truth is discovered during implementation, prefer a
  `CacheLayerMixin` implementation so this unit matches the HuggingFace per-layer cache role.
- If the HuggingFace-provided sliding-window cache continues to satisfy the local path cleanly,
  it should be used directly as `.sliding_window_cache` rather than reimplemented.
- The MoSRAH side should be owned through the 6.A cache unit directly as `.mosrah_cache`.
- This unit should prefer straightforward ownership and transparent dispatch over abstraction
  unification. If a proposed shared interface would hide materially different semantics between
  the two cache paths, it should be rejected.
- If a HuggingFace per-layer method has a truthful composite meaning here, it should be
  implemented at this boundary and forwarded appropriately to the owned sub-caches.
- If a HuggingFace per-layer method has no truthful composite meaning here, explicit
  `NotImplementedError()` is preferable to a misleading partial implementation.

### 6.C SHRAM cache

**Responsibility:** Define and verify the top-level cache object for the full SHRAM model. This
unit owns the per-layer `ShramLayerCache` objects, presents them through the HuggingFace `Cache`
interface, and transparently coordinates model-wide cache behaviors across all decoder layers.

**Context of Correctness**

HuggingFace wants the model-wide cache object to exist as a top-level `Cache`, while the actual
SHRAM cache responsibilities live one layer lower in the per-layer `ShramLayerCache`. For this
reason, the correct top-level cache unit is a `ShramCache` that owns one `ShramLayerCache` per
decoder layer and transparently forwards model-wide cache operations across them.

This unit should behave externally as the single cache object for the model, while internally
preserving the layer structure required by both HuggingFace and the SHRAM architecture. In
particular, any model-wide cache concept that has a truthful scalar meaning should be exposed
here, while anything that does not should fail explicitly. The most important example is total
sequence length: this belongs at the top-level cache boundary, and is derived from the
sliding-window side because that path tracks the full causal sequence length while the MoSRAH
path is ragged and does not.

**Invariants this unit must satisfy:**
- The unit defines the top-level cache object for the SHRAM model. It owns the full collection
  of per-layer `ShramLayerCache` objects and no lower-level cache state lives directly at this
  boundary.
- The cache fulfills the HuggingFace top-level `Cache` role.
- The cache contains exactly one `ShramLayerCache` per decoder layer.
- The cache behaves externally as the single cache object for the model while internally
  preserving the per-layer cache structure.
- Any top-level cache operation with a truthful model-wide meaning is implemented here and
  transparently forwarded across the owned layer caches as needed.
- Reset and reorder behavior are implemented model-wide by transparently applying them across all
  owned `ShramLayerCache` objects.
- The authoritative scalar sequence-length concept is implemented at this boundary.
- That scalar sequence length is derived from the sliding-window side of the layer caches, since
  that path tracks total causal sequence length while the MoSRAH sparse path does not.
- All necessary HuggingFace top-level cache methods have been evaluated and implemented to a
  locally sane standard.
- Any HuggingFace top-level cache method whose semantics do not truthfully apply at the SHRAM
  model-cache boundary instead raises `NotImplementedError()`.
- The unit is implemented for very rapid usage.
- Tests and design for this unit are themselves structured so the unit can be trusted as a
  verified black box by later model and generation units.

**Tests**
- Verify the unit owns exactly one `ShramLayerCache` per decoder layer.
- Verify the unit behaves externally as a single top-level cache object for the model.
- Verify reset behavior clears all owned layer caches through the top-level boundary.
- Verify reorder behavior applies consistently across all owned layer caches through the
  top-level boundary.
- Verify any supported HuggingFace top-level cache operation dispatches correctly across the
  owned layer caches.
- Verify the top-level scalar sequence length is exposed correctly.
- Verify the scalar sequence length is sourced from the sliding-window side rather than the
  MoSRAH sparse side.
- Verify unsupported top-level operations fail explicitly rather than partially mutating only a
  subset of layer caches or returning misleading results.
- Verify required HuggingFace top-level cache behaviors are implemented sanely.

**Preliminary implementation strategy:**
- If no contradictory ground truth is discovered during implementation, prefer a standard
  HuggingFace `Cache` implementation whose entries are the per-layer `ShramLayerCache` objects.
- If a truthful scalar sequence length is needed, it should be sourced from the local
  sliding-window cache path rather than inferred from MoSRAH sparse occupancy.
- Top-level reset and reorder should be implemented as transparent passes over all owned layer
  caches.
- This unit should prefer straightforward ownership and transparent forwarding over introducing
  additional abstraction layers.
- If a HuggingFace top-level cache method has a truthful model-wide meaning here, it should be
  implemented and forwarded appropriately.
- If a HuggingFace top-level cache method has no truthful model-wide meaning here, explicit
  `NotImplementedError()` is preferable to a misleading partial implementation.

### 6.Final — Generation Strategy Support

During 6.B/6.C review it was identified that `batch_repeat_interleave` (beam search init) and
`batch_select_indices` (contrastive search) were incorrectly raising `NotImplementedError`.
Both were straightforward extensions of the existing `reorder_cache` pattern and were
implemented across the full stack: `MoSRAHCache`, `SlowMoSRAHCache`, `ShramLayerCache`, and
`ShramCache`. `crop` (speculative decoding rollback) was deferred — its semantics for the
MoSRAH ragged cache are a research design question and were not resolved here.

---

### Unit 7 — Local Sliding-Window Attention Module

**Responsibility:** Define and verify the local sliding-window attention path `h_l` for one SHRAM decoder layer.

**Context of Correctness**

This unit is the short-range sliding-window dot-product attention path inside SHRAM. It needs to be able to allow operation over the 64k sequence lengths during the experiment. The layer must support caching for later experimentation and have positional encoding. Because it is a local attention formulation it should not adjust its positional (RoPE) formulation when YaRN extrapolation occurs. It has to be causal as this is an autoregressive decoder context. It must be compatible with the broader SHRAM layer as well.

**Invariants this unit must satisfy:**
- The unit defines `h_l` for one SHRAM decoder layer only.
- The unit implements the local short-range dot-product attention path inside SHRAM.
- The unit implements causal sliding-window attention.
- The unit supports the experimental long-sequence regime required by the project.
- The unit accepts and uses the `.sliding_window_cache` belonging to its corresponding
  `ShramLayerCache`.
- The unit supports cached execution for later experimentation.
- The unit implements positional encoding for the local path.
- The local path's positional encoding uses `default` RoPE mode and does not respond to YaRN
  dilation or long-context scaling changes elsewhere in the model.
- The unit returns outputs in `(B, N, d)` form compatible with downstream SHRAM composition.
- The unit is compatible with the broader SHRAM layer and cache architecture.
- The unit is implemented for very rapid usage.
- Tests and design for this unit are themselves structured so the unit can be trusted as a
  verified black box by later SHRAM-layer assembly.

**Tests**
- Verify output shape is correct.
- Verify tokens outside the local window do not contribute to attention.
- Verify causal ordering is preserved within the active local window.
- Verify the unit consumes and updates the provided `.sliding_window_cache` correctly.
- Verify repeated cached forward passes produce correct accumulated local-cache behavior.
- Verify local positional encoding is constructed in `default` mode.
- Verify local positional behavior does not change when YaRN / long-context scaling changes
  elsewhere in the model configuration.
- Verify required cache- and backend-facing behaviors are implemented sanely.
- Verify the returned output remains compatible with later SHRAM composition.

**Preliminary implementation strategy:**
- If no contradictory ground truth is discovered during implementation, the local path should
  receive and consume the layer-local `.sliding_window_cache` directly rather than re-resolving
  broader cache ownership internally.
- If FlexAttention continues to satisfy the required sliding-window, causal, and long-sequence
  behavior cleanly, it should be the preferred backend for this unit.
- If the current `RotaryEmbedding` design continues to support `default` mode that ignores
  dilation, that mode should be used for the local path.
- If cached and uncached execution diverge in behavior, correctness takes priority and the
  divergence should be surfaced explicitly rather than hidden behind fallback logic.
- If the chosen backend introduces a limitation that breaks any of the above invariants, that
  limitation should be surfaced and the backend choice reconsidered explicitly rather than worked
  around silently.
---
### Unit 8 — Expert Packing and Unpacking

**Responsibility:** Define and verify the packing and unpacking subsystem that converts routed
token-choice state into expert-choice packed state for BEA, and then restores token-choice
ordering afterward.

**Context of Correctness**

This unit sits at the conversion boundary between routing and BEA. Routing produces token-choice state, while BEA consumes expert-choice packed tensors. The packing and unpacking process is not merely an implementation convenience here: the paper specifies the algorithmic structure in detail, and this portion of the system is especially sensitive to being implemented incorrectly. In particular, causal ordering inside expert buckets, the packed position tensor used by RoPE, the active-token mask, and the inverse mapping back to token-choice order are all load-bearing. For that reason, this unit is more constrained than many others: the paper’s stated algorithm is part of the correctness boundary and packing  packing/unpacking is not an open design space.

**Invariants this unit must satisfy:**
- Expert packing and unpacking are implemented as specified in Appendix A / Architecture Details of the paper, especially the expert packing/unpacking tensor implementation and its auxiliary tensors.
- An auxiliary setup_packing function is used as described in the paper to prepare the auxiliary data and make implementation modular.
- Packing converts routed token-choice state into expert-choice packed state for BEA, and
  unpacking restores token-choice ordering afterward.
- Stable sort is used wherever the paper requires it to establish expert-major ordering.
  Unstable sort is not acceptable.
- The packed representation includes:
  - packed hidden states for BEA
  - packed original-token positions for downstream RoPE use
  - an active-token mask distinguishing real packed tokens from padding
  - the inverse-ordering context required to restore token-choice order exactly
- The active-token mask has the correct cardinality and correctly distinguishes active packed
  tokens from padding.
- Active packed tokens are left-justified within each expert slot.
- Packed positions preserve the original sequence positions in the packed order required by the
  paper.
- Unpacking is the exact inverse of packing on active entries.
- `unpack(pack(x, selected_heads))` recovers the original active token copies at the correct
  token-choice locations.
- Padding introduced during packing remains padding and does not become active data during
  unpacking.
- The unit exposes enough activity information for later attention and inference code to tell
  which regions correspond to real routed tokens and which correspond to padding.

**Tests**
- Verify the implemented packing/unpacking path matches the paper-specified algorithmic behavior.
- Verify stable-sort-based packing preserves causal order within each expert bucket.
- Verify a deliberately unstable-sort alternative would fail the causal-order test.
- Verify `active_mask` has the correct entry count and correctly identifies active tokens.
- Verify active packed tokens are left-justified.
- Verify packed positions match original sequence positions in packed order.
- Verify unpacking restores outputs to token-choice order with the correct shape.
- Verify round-trip identity on active entries.
- Verify padding regions remain zero/inactive through unpacking.

**Preliminary implementation strategy:**
- The paper should be treated as the primary algorithmic specification for this unit.
  otherwise defer to the paper rather than re-specifying the full algorithm here.
- If an implementation choice would alter the packed ordering, position tensor semantics,
  activity-mask semantics, or inverse mapping described in the paper, it should be rejected.
- If a more efficient implementation is introduced, it must preserve exactly the externally
  visible behavior specified by the paper.

### Unit 8 — Expert Packing and Unpacking

**Invariants this unit must satisfy:**
- Expert packing converts `selected_heads` (B, N, K) + x (B, N, d) into `packed_hidden`
  (B, L, T, d), `packed_positions` (B, L, T), `active_mask` (B, L, T boolean) as specified
  in Appendix A.Expert Packing.
- Stable sort is used to produce `expert_order_idx` — not argsort without stable=True.
  Unstable sort silently violates causal ordering of tokens within each head's packed
  representation.
- `active_mask` has exactly B×N×K true entries. Active tokens are left-justified within each
  head slot.
- `packed_positions` contains original sequence positions (0..N-1) of the packed tokens.
- `token_order_idx` (inverse permutation) is retained for unpacking.
- Expert unpacking is the exact inverse: given y (B, L, T, d) and the packing context
  (`token_order_idx`, `active_mask`), produces `unpacked_output` (B, N, K, d).
- Round-trip identity: unpack(pack(x, selected_heads)) recovers x at all original positions.
- Tests verify: stable sort causality (a test that fails with unstable sort),
  `active_mask` entry count, `packed_positions` correctness, round-trip identity, padding
  is zero.
- Inference modes will need to know what sections are padding vs active once unpacked. 

**Paper refs:** Appendix A.Expert Packing and Unpacking.

---

### Unit 9 — Bottlenecked Ensemble Attention (BEA)

BEA is a pure attention operator: given a packed expert-choice tensor, a position tensor, a
mask, and an optional cache, it projects Q/K/V, applies RoPE, writes K̃ and V to the cache if
present, reads back accumulated K̃/V, and attends. BEA does not compute positions — the caller
(Unit 10.B) supplies the position tensor P. BEA does not choose the RoPE mode — that is
MoSRAH's responsibility. This keeps BEA's contract simple and independently testable.

Storing post-RoPE K̃ in the cache (rather than raw K) means cached keys are already rotated;
subsequent reads use them directly without re-applying RoPE. This is a correctness requirement:
applying RoPE again to a cached K̃ would corrupt positional encoding.

**Invariants this unit must satisfy:**
- Interface: `forward(packed_hidden, position_ids, active_mask, cache=None, layer_idx=None)`
  where `packed_hidden` is `(B, L, T, d)`, `position_ids` is `(B, L, T)` int,
  `active_mask` is `(B, L, T)` bool. Returns `y` of shape `(B, L, T, d)`.
- `W_Q`, `W_K`, `W_V` shape `(L, d, u)`; `W_O` shape `(L, u, d)`. Independent parameters
  per head — no weight sharing across heads. `u = d / K` (mosrah_head_dim from config).
- BEA constructs its own `RotaryEmbedding(mode="yarn", head_dim=config.head_dim,
  theta=config.mosrah_rope_theta, initial_seq_length=config.training_sequence_length,
  dilation=config.scale, alpha=config.alpha, beta=config.beta)`. The returned `A_rope`
  scales attention logits before softmax.
- RoPE is applied to Q and K using `position_ids`. The result K̃ is post-RoPE.
- If `cache is not None`: `cache.mosrah_cache.write(layer_idx, K̃, V, active_mask)`, then
  `K̃_all, V_all, full_mask = cache.mosrah_cache.read(layer_idx)`. Attention uses
  `K̃_all`, `V_all`, `full_mask`.
- If `cache is None`: attention uses K̃, V, `active_mask` directly.
- Causal masking is a triangular mask over the T (packed) dimension.
- Padded positions (`active_mask == False`) do not contribute to attention output.
- Tests verify: output shape; head independence (perturbing one head's weights changes only
  that head's output); RoPE passthrough (different position tensors produce different outputs);
  padding positions do not influence real-token outputs; causal masking holds; with cache,
  accumulated K̃/V grow correctly across calls and are used in attention.

**Paper refs:** Appendix A.BEA, Appendix B.RoPE Mechanics.

---

### Unit 10.A — MoSRAH Position Computation (Blocker for 10.B)

BEA receives a position tensor P of shape `(B, L, T)` from its caller. This unit specifies
exactly how P is computed under both rope modes and both cache states. It is a blocker for
Unit 10.B — MoSRAH assembly must not begin until this computation is specified and verified.

There are two rope modes (config-selected) and two cache states (training vs inference),
giving four cases that reduce to two formulas:

**Main-sequence mode:** `P = J` always. `J` is the packed original-token position tensor
produced by expert packing (Unit 8), shape `(B, L, T)`. Cache state is irrelevant.

**Semantic-sequence mode:** `P = base + arange(T)` where:
- `base = 0` if `cache is None` (training). Positions are 0, 1, 2, ... within each head slot.
- `base = cache.mosrah_cache.get_expert_lengths(layer_idx)` if `cache is not None` (inference).
  Shape `(B, L)`, broadcast to `(B, L, T)`. Positions are offset by the current slot
  occupancy so the new token(s) continue the semantic sequence already in the cache.

`get_expert_lengths()` must be read **before** BEA calls `write()`, since `write()` increments
the counts. The call order in Unit 10.B is: compute P (this unit) → call BEA (which writes,
then reads). This ordering is the invariant — it must not be violated.

**Invariants this unit must satisfy:**
- Main-sequence mode returns `J` unchanged regardless of cache state.
- Semantic mode, no cache: returns `arange(T)` broadcast to `(B, L, T)`.
- Semantic mode, with cache: returns `get_expert_lengths(layer_idx).unsqueeze(-1) + arange(T)`
  broadcast to `(B, L, T)`.
- Position computation reads `get_expert_lengths()` before any `write()` occurs in this
  forward pass.
- Tests verify: main-sequence returns J; semantic no-cache returns slot indices; semantic
  with-cache returns count-offset positions; bulk update (T > 1) produces contiguous positions;
  positions read before write confirmed by populating cache and asserting offset is correct.

---

### Unit 10.B — MoSRAH Sparse Path

**Invariants this unit must satisfy:**
- MoSRAH forward: `(selected_heads, routing_probs) = Router(x)`; `packed_hidden, J, active_mask
  = Pack(x, selected_heads)`; `P = PositionComputation(J, cache, layer_idx)` (Unit 10.A);
  `y = BEA(packed_hidden, P, active_mask, cache, layer_idx)`; `unpacked_output = Unpack(y)`;
  `o = sum_k unpacked_output_k * routing_probs_k`. Matches Algorithm 1 in the paper.
- Weighted reduction uses `routing_probs` (unbiased, renormalized) — not biased scores,
  not `selected_heads` indices.
- `load_balance_loss` from Router is propagated through MoSRAH's return.
- MoSRAH receives `cache` (a `ShramCache` or `None`) and `layer_idx`. It accesses
  `cache.mosrah_cache` directly; it does not call `cache.update()`.
- Input/output shape: `(B, N, d)` → `(B, N, d)`.
- Tests verify: output shape; gradient flows through `routing_probs` to router weights;
  `load_balance_loss` present; weighted reduction correct; cache accumulates correctly
  across calls.

**Paper refs:** §3 Design, Algorithm 1, Appendix A throughout.

---

### Unit 11 — SHRAM Hybrid Layer

**Invariants this unit must satisfy:**
- SHRAM forward: H(x) = h_l(x) + h_s(x), where h_l is the local sliding-window module
  (Unit 7, verified black box) and h_s is MoSRAH (Unit 10.B, verified black box).
- h_l and h_s have fully independent parameters. Their outputs are summed.
- load_balance_loss from h_s is propagated through the hybrid layer's return.
- Output shape: (B, N, d) — same interface as GroupedQueryAttention it replaces.
- Tests verify: output shape, independence of paths (zeroing h_l parameters leaves only
  h_s contribution and vice versa), load_balance_loss is present in the return.

**Paper refs:** §3 Design Success Conditions.

---

### Unit 12 — DecoderLayer Update

**Invariants this unit must satisfy:**
- DecoderLayer uses SHRAM hybrid layer in place of GroupedQueryAttention.
- load_balance_loss returned by the SHRAM layer is propagated through DecoderLayer's output.
- The pre-norm + residual structure is unchanged: x' = x + SHRAM(RMSNorm(x));
  x'' = x' + FFN(RMSNorm(x')). This structure transfers unchanged per Appendix A.
- Tests verify: output shape unchanged, load_balance_loss present in output, structure holds.

---

### Unit 13 — ShramModel Update

**Invariants this unit must satisfy:**
- ShramModel collects load_balance_loss from all decoder layers and includes an aggregated
  scalar in the output dict.
- The existing output dict keys (last_hidden_state, past_key_values, hidden_states) are
  preserved with unchanged semantics.
- Tests verify: load_balance_loss present in output dict, aggregation is correct (not just
  the last layer's value — it must reflect all layers), existing output unchanged.

---

### Unit 14 — ShramForCausalLM Update

**Invariants this unit must satisfy:**
- `load_balance_loss` is accessible from the forward output so the training loop can weight and
  apply it. The consumer is responsible for the weight (1e-2 in the paper; job.md Architecture
  § confirms this is the training loop's responsibility).
- `ShramForCausalLM` overrides `_prepare_cache_for_generation` to instantiate `ShramCache`
  with parameters derived from model config. `generate()` called without an explicit
  `past_key_values` produces a `ShramCache` internally. A caller who passes an explicit
  `ShramCache` has it used unchanged.
- `use_cache=True` is supported. The `past_key_values` flowing through the model is always a
  `ShramCache` — never a plain `DynamicCache`.
- Tests verify: `load_balance_loss` in output; `generate()` without explicit cache produces
  `ShramCache`; `generate()` with explicit `ShramCache` uses it unchanged; KV cache
  accumulates correctly across decode steps.

**Research notes:**
- `_prepare_cache_for_generation(generation_config, model_kwargs, generation_mode,
  batch_size, max_cache_length)` is the override point. It writes to
  `model_kwargs["past_key_values"]` as a side effect.
- GenerationMixin computes `cache_position` as `torch.arange(N, N+M)` for M new tokens
  when N tokens are already cached.

---

### Unit 15 — upload_to_hub.py

**What:** Adapt the upload script for the SHRAM model type. Update class names, model type
string, and model card content. Register `ShramConfig` and `ShramForCausalLM` with the AutoClass
API and push all model files to the Hub.

**Invariants this unit must satisfy:**
- A freshly instantiated model can be round-tripped: upload → `from_pretrained` → forward pass.
- The model card accurately describes the SHRAM architecture.
- No weights are uploaded. No checkpoint is assumed.

---

### Unit 16 — Documentation

**What:** Write `documentation.md` covering design decisions, deviations from the paper, and
limitations. Update `README.md` with accurate architectural details. Record every open decision
resolved during implementation and the rationale for each.

**Invariants this unit must satisfy:**
- Every open decision resolved during implementation is recorded with its rationale.
- Limitations are documented explicitly, including any known train/inference mismatches.
- The model card accurately describes the SHRAM variant.

---

### Unit 17 — End-to-End Tests

**What:** Full-stack smoke tests: instantiate from config, run a training step, verify loss
decreases. Include load-balance loss in the training step. Include network tests for the Hub
round-trip.

**Invariants this unit must satisfy:**
- The model can be instantiated from a config, run a forward pass, compute loss, and backpropagate
  without error.
- The load-balance loss is accessible from the forward output and participates in the backward pass.
- Network tests verify the Hub round-trip.

---

### Unit 18 — Final Audit

**What:** Review every file in `src/shram/` against the invariants in `job.md`. Verify no
hardcoded values, no missing documentation, no gaps between tests and intent. Apply the
close-the-testing-gap rule to any defect found.

**Invariants this unit must satisfy:**
- Every invariant in job.md is satisfied and has a corresponding test.
- No file has hardcoded architectural parameters.
- All documentation standards are met.
- Every decision recorded in this plan has a corresponding entry in `documentation.md`.
