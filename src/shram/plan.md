# Implementation Plan: advanced-transformers-lib — SHRAM

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
- [X] Unit 6.A.A — MoSRAHCache: custom buffer storage and HF Cache protocol
- [X] Unit 6.A.B — MoSRAHCache: vectorized scatter update
- [ ] Unit 6.A.C — MoSRAHCache: test audit and trust verification
- [ ] Unit 6.B — ShramCache: HF interface shim holding both sub-caches
- [ ] Unit 6.C — ShramCache generation pipeline wiring dispatch as blocker via _prepare_cache_for_generation
- [ ] Unit 6.D — MoSRAH sequence position decoder: formal specification and blocker registration
- [ ] Unit 7 — Local sliding-window attention module (h_l)
- [ ] Unit 8 — Expert packing and unpacking: permutation machinery, padding, masks
- [ ] Unit 9 — Bottlenecked Ensemble Attention (BEA): per-head attention on packed tensors
- [ ] Unit 10 — MoSRAH sparse path: routing → packing → BEA → unpacking → weighted reduction
- [ ] Unit 11 — SHRAM hybrid layer: assemble H(x) = h_l (Unit 7) + h_s (Unit 10)
- [ ] Unit 12 — DecoderLayer: replace attention sublayer, propagate load_balance_loss
- [ ] Unit 13 — ShramModel: aggregate load_balance_loss in output
- [ ] Unit 14 — ShramForCausalLM: expose load_balance_loss; KV cache resolution
- [ ] Unit 15 — upload_to_hub
- [ ] Unit 16 — documentation
- [ ] Unit 17 — end-to-end tests
- [ ] Unit 18 — final audit

---

## Status

**Current state:** Units 5.A, 5.B, 5.C, 5.C.1, 6.0, 6.A.A, and 6.A.B complete. Unit 6.A.C (test audit and trust verification) is next. Network tests deselected (Hub repo not yet created).

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
  `get_expert_lengths(layer_idx)` to obtain the counts and are responsible for masking.
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

### Unit 6.0 — Cache Architecture: Situation, Decisions, Folder Refactor

**Situation:** The existing `mosrah_cache.py` was written before the position tracking
architecture was understood. The design was, however, done incorrectly. 

**Refactor**:

All caches and cache code should be placed in src/shram/model/cache instead. Likewise tests should be moved to tests/shram/model/tests

*Why several cache classes exist:*
SHRAM has two attention paths with fundamentally different KV semantics. h_l processes
every token with a sliding window; MoSRAH processes only routed tokens per head,
unboundedly. No single cache class can serve both correctly. ShramCache exists because
HF infrastructure requires one object in `past_key_values` — it is a shim, not a
functional cache. These three roles are genuinely distinct and must not be collapsed. It has been designed to 

*Generation, Training, and Sequences*

Sequence position was an enormous component that had to be carefully handled in order to ensure rope worked for all layers in all modes during both training and evaluation.

Huggingface will natively pass in a tensor of positional ids during evaluation which is normally compatible with rope. However, MoSRAH can operate in an alternative mode in which only the sequence length inside the headed dimensions mattered. This required handling getting the lengths of each head in order to be able to decode what rope position each would be in in these modes. 

Both training and evaluation would need to handle these challenges. As a result, this decoding code is now planned to be placed on the MoSRAH unit. It handles whatever positional shifting is needed internally, and the cache system just needs to display the perntient information. This allows the same underlying process to handle both issues just by modifying where a cumsum starts at.

**Invariants this unit must satisfy:**
- `src/shram/model/cache/` exists as a package with `__init__.py`. All cache classes live under
  `src/shram/model/cache/`.
- `tests/shram/model/cache/` mirrors the cache package structure.

**Notes**

- The semantic sequence mode can be implemented by the following sequence of vectorized operation. Make a batch x sequence_length x number_of_total_heads boolean tensor filled with false. Scatter by vector indexing along the total heads dimension the selected heads tensor (I) a value of true. Now, cumsum along the sequence length direction. This gives the number of times this block of operations has routed to this head in this unit. Subtract one from this and add it to the length of each head sequence, then gather using I from this tensor array for the final rope positions. This does not end up in MoSRAH cache but downstream
- 

### Unit 6.A.A — MoSRAHCache: Custom Buffer Storage and HF Cache Protocol

MoSRAH routes each input token to K of L expert heads. In the generation setting this means
the KV cache must accumulate keys and values per expert head independently — only the K heads
selected for the current token are updated at each decode step, while the remaining L−K are
untouched. The central problem is that this produces a ragged distribution of token counts
across (batch, head) slots: different batch items may route different numbers of tokens to the
same head within a single forward pass, and different heads accumulate at entirely different
rates across decode steps. `DynamicCache` assumes a uniform sequence dimension per slot and
concatenates along it; there is no mechanism for it to track valid occupancy independently per
(batch, head). Forcing MoSRAH's ragged structure into `DynamicCache` would require external
bookkeeping that bypasses the cache entirely — which is not acceptable.

The solution is a custom buffer: `self.keys` and `self.values` each of shape `(B, L, T_max, u)`
per layer, plus `self.counts` of shape `(B, L)` — an integer tensor tracking the valid
occupancy of each (batch, head) slot. `T_max` is the shared current capacity per layer, doubled
when any slot overflows. Everything in the buffer beyond the count for a given slot is junk;
consumers use `get_expert_lengths(layer_idx)` to know what is valid and are responsible for masking.

This unit establishes the storage foundation and all HF Cache protocol methods. `update()`
raises `NotImplementedError` — the scatter logic that populates the buffer is the sole
responsibility of Unit 6.A.B. The split exists because storage correctness is independently
verifiable, and because the scatter is complex enough to be its own verified unit.

**Responsibility:** Provides the `(B, L, T_max, u)` buffer structure and `(B, L)` counts per
layer, `get_expert_lengths()`, and all coordination operations (`reset`, `reorder_cache`). Does
NOT implement the scatter update.

**Invariants this unit must satisfy:**
- `MoSRAHCache` subclasses `transformers.cache_utils.Cache`. `Cache.__init__` is called with
  `layers=[]`; `self.layers` is unused.
- `__init__(num_mosrah_heads: int, head_dim: int)` stores L and u. No tensors are allocated
  until the first `update()` call — lazy allocation ensures device placement matches the
  incoming tensors.
- Per layer, `self.keys` and `self.values` are `(B, L, T_max, u)` float tensors and
  `self.counts` is a `(B, L)` integer tensor. All three live on the same device. Per-layer
  storage is allocated lazily and independently — earlier layers do not force later layers to
  allocate.
- `get_seq_length()` raises `NotImplementedError`. No single sequence length meaningfully
  represents a cache where each (batch, head) slot accumulates independently.
- `get_expert_lengths(layer_idx: int) -> torch.Tensor` returns the counts tensor of shape
  `(B, L)` for that layer. This is the authoritative per-head occupancy consumed by BEA for
  attention masking and by the position decoder (Unit 6.D) for semantic-sequence position
  computation. Returns a zero tensor of the correct shape if the layer has not yet been
  allocated.
- `reset()` clears all per-layer storage. After reset, `get_expert_lengths()` returns zeros.
- `reorder_cache(beam_idx)` reorders dim 0 of both the key/value buffers and counts across all
  allocated layers. Beam search must apply atomically or buffer contents and occupancy counts
  will diverge.
- `update()` raises `NotImplementedError`.
- Tests verify: `Cache` subclass; `get_seq_length` raises; `get_expert_lengths` returns `(B, L)`
  shape and correct values after manual count and buffer assignment; `reset` clears all layers;
  `reorder_cache` correctly permutes the batch dimension of both buffers and counts; `update`
  raises `NotImplementedError`.

**Preliminary implementation notes:**
- Initial `T_max` on first allocation: a small power of two (e.g. 64) is preferable to
  starting at 1 to avoid repeated reallocation during prompt processing. Decide during
  implementation.
- Buffer expansion: when any `counts[b, h]` would exceed the current `T_max` for a layer,
  allocate new `(B, L, 2*T_max, u)` tensors, copy old data, replace. `T_max` is shared
  across all (batch, head) slots within a layer — the entire layer buffer doubles when any
  slot overflows.
- For tests that exercise `reset` and `reorder_cache`, populate buffers and counts directly
  (e.g. `cache.counts[layer] = tensor; cache.keys[layer] = tensor`) rather than calling
  `update()`, since `update()` is a placeholder in this unit.

---

### Unit 6.A.B — MoSRAHCache: Vectorized Scatter Update

The buffer established in 6.A.A has no mechanism for populating it — `update()` raises
`NotImplementedError`. This unit implements the scatter operation that takes the token-choice
routing output `(I, K_states, V_states)` and writes each token's key and value into the correct
buffer slot at the correct position.

The correctness constraint is causal ordering: for a given (batch, head) slot, tokens must
appear in the same order they occupied the original sequence. This is the same invariant that
makes stable sort a correctness requirement in expert packing (Unit 8). Violating it here
corrupts the accumulated KV history — making the attention computed in BEA non-causal in a way
that will not raise an error and may not surface until evaluation. The scatter uses the cumsum
construction described in the Unit 6.D position decoder to derive within-slot target indices
without a loop over sequence positions: a `(B, N, L)` boolean head-selection mask is formed
from `head_idx`, cumulatively summed along N to give the within-slot index for each
(batch, token, chosen-head) triple, offset by the pre-update counts to give the absolute
buffer position.

The test strategy makes correctness visible: a non-vectorized reference implementation is
written first — a Python loop over `(b, n, k)` that directly indexes into the buffer and is
obviously correct by inspection. The vectorized implementation is then validated against it on
all test cases. The reference is not a throwaway sketch; it is the ground truth that licenses
trust in the production implementation. It can be used later in how 6.D is dispatched as well.

**Responsibility:** Implements `update(layer_idx, head_idx, key_states, value_states)` and
triggers buffer expansion when needed. Callers receive the full `(B, L, T_max, u)` buffers as
the return value; they obtain the counts needed to mask padding by calling
`get_expert_lengths()` separately. NOT responsible for routing, BEA attention, or position
computation.

**Invariants this unit must satisfy:**
- `update(layer_idx: int, head_idx: torch.Tensor, key_states: torch.Tensor,
  value_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]` where `head_idx` is
  `(B, N, K)` int, `key_states` and `value_states` are `(B, N, K, u)` float.
- Returns `(self.keys[layer_idx], self.values[layer_idx])` — the full `(B, L, T_max, u)`
  buffers after the update.
- After `update()`, `self.counts[layer_idx][b, h]` equals the total number of tokens written
  to head `h` for batch item `b` in that layer, accumulated across all calls since last reset.
- Causal ordering: for a given `(b, h)`, tokens appear in the buffer in the order they
  appeared along the N dimension of `head_idx`.
- Buffer expansion: if any slot would overflow `T_max` after this update, the buffer is doubled
  before the scatter proceeds.
- `update()` and the reference implementation produce identical `(keys, counts)` on all test
  inputs.
- Tests verify: single-token round-trip; multi-call accumulation; scatter ordering correctness
  (vectorized matches reference); sparse routing (only a subset of heads updated per step,
  others unchanged); uneven per-head counts across batch items; buffer expansion triggers
  correctly and preserves existing data.

**Research notes:**
- Cumsum scatter for within-slot positions: form `(B, N, L)` bool mask by scattering True at
  positions given by `head_idx`; cumsum along N gives per-(b, n, h) cumulative count; subtract
  1 for the zero-based within-slot index. Add `self.counts[layer_idx][b, head]` (the
  pre-update count) for the absolute buffer index. This is the same construction used in
  Unit 6.D for semantic-sequence positions.
- Reference implementation (write first, use as test oracle): for each `(b, n, k)`, execute
  `h = head_idx[b, n, k]; pos = counts[b, h]; keys[b, h, pos, :] = key_states[b, n, k, :];
  counts[b, h] += 1`. Run this on the same inputs as the vectorized version and assert
  identical results.
- `torch.bincount` only handles 1D input. To count tokens per head per batch item from the
  flattened `(B, N*K)` head index tensor, a strided multidimensional variant is needed — see
  the paper's footnote on `bincount_multidimensional` for the approach.

### Unit 6.A.C — MoSRAHCache: Test Audit and Trust Verification

Units 6.A.A and 6.A.B produce a working cache. They are worth nothing until this unit is
complete. A cache implementation that runs and passes tests it wrote for itself is not a
verified cache — it is code that confirms its own behavior, which is a different and weaker
thing. The scatter operation is novel, the storage design is custom, and the correctness
invariants are subtle: causal ordering, per-batch ragged routing, expansion under load,
multi-layer independence. Any one of these could be violated by an implementation that passes
a test suite written by the same author who wrote the implementation, under the same
assumptions, with the same blind spots.

This unit is not about adding test cases. It is about asking, for every invariant in 6.A.A and
6.A.B: do the existing tests actually enforce this, or do they merely confirm that the
implementation does what it does? That is a harder question and it requires treating the tests
themselves as objects of scrutiny, not as evidence of correctness. Where gaps exist — and there
will be gaps — they must be closed. Where tests pass for the wrong reasons, they must be
corrected. Where the space of valid inputs has not been adequately explored, it must be.

The agent handling this unit must think through what it would take to convince a skeptical
reader that the cache is correct — not merely that it passes its test suite.

**Responsibility:** Audit the test suites produced in 6.A.A and 6.A.B. Identify every gap
between what the tests enforce and what correctness requires. Close those gaps. Verify that the
reference implementation in 6.A.B is genuinely independent — not sharing any code path with
the vectorized implementation that could allow both to be wrong in the same way. Add lifecycle
tests that require a working `update()` and therefore could not exist in 6.A.A. Produce a
test suite that a skeptical reviewer could trust.

**Known concrete cases that must be present:**
- `reset()` with populated cache: update multiple layers and multiple heads, call reset, verify
  `get_expert_lengths()` returns zeros for all layers and that subsequent updates start from
  position 0.
- `reorder_cache()` with real data: populate via `update()`, apply a non-trivial batch
  permutation, verify that both buffer contents and counts are reordered consistently —
  not just that the shapes are correct.
- Multi-layer independence: writing to layer 0 does not affect layer 1, and vice versa. Verify
  content, not just shape.
- Buffer expansion preserves data: trigger an expansion (fill a slot past T_max), verify that
  all previously written data is intact and correctly accessible after the resize.
- Uneven batch routing: batch item 0 and batch item 1 route to different heads in the same
  call. Verify each (batch, head) slot contains exactly the tokens written to it, with no
  bleed between batch items.

**Abstract variables the agent and user much reach consensus on**
- Is the reference implementation in 6.A.B truly independent of the vectorized one? If they
  share any utility function or data structure, find and eliminate the coupling.
- What happens when the same head is selected multiple times within a single update call
  (e.g. head_idx[b, n1, k1] == head_idx[b, n2, k2] == h for different (n1,k1), (n2,k2))?
  Does the scatter handle this correctly and in causal order?
- Are there edge cases at the boundaries of T_max expansion — e.g. filling exactly to T_max,
  then adding one more token — where off-by-one errors could corrupt data?
- Does `get_expert_lengths()` return a copy or a view? If a view, can the caller mutate it and
  corrupt the cache state? Should it return a copy?
- After `reorder_cache()`, does subsequent `update()` correctly target the reordered batch
  slots, or does the reordering only affect retrieval?

---

### Unit 6.B — ShramCache

**Responsibility:** Satisfies the HF Cache protocol as the single object in
`past_key_values`. Holds both sub-caches as named attributes. Coordinates protocol
operations (reset, reorder_cache) atomically across both. Reports total tokens
processed via get_seq_length(). NOT responsible for: attention computation, position
semantics, routing decisions, or knowing anything about modes.

**Invariants this unit must satisfy:**
- `ShramCache` subclasses `transformers.cache_utils.Cache`.
- `self.local_cache` is a `DynamicSlidingWindowCache`; `self.mosrah_cache` is a
  `MoSRAHCache`. Both are accessible as named attributes. Attention code accesses
  them directly; HF infrastructure interacts only with `ShramCache`.
- `update()` raises `NotImplementedError`. The two sub-caches have incompatible
  update semantics; there is no correct single-entry-point routing. This is a
  deliberate design boundary, not an oversight.
- `get_seq_length()` delegates to `local_cache`. Conceptually this returns total
  tokens processed — the quantity GenerationMixin needs to compute cache_position
  for the next step. Implemented via local_cache because h_l processes every token
  without exception, making its length equal to total tokens processed.
- `reset()` resets both sub-caches.
- `reorder_cache(beam_idx)` reorders both sub-caches — beam search must apply
  atomically to both or the two paths diverge on different beam hypotheses.
- Tests verify: local_cache and mosrah_cache accessible as attributes; update()
  raises NotImplementedError; get_seq_length() reflects local cache length; reset()
  clears both; reorder_cache() reorders both.

---

### Unit 6.C — ShramCache: Generation Pipeline Wiring

**Responsibility:** Wire up plan features for appropriate revision unit to ensure ShramCache is always what our and huggingface infrastructure receives in
`past_key_values` during generate() — never a plain DynamicCache. NOT responsible
for: any caching behavior beyond ensuring the correct cache type is instantiated. Verify that strategy will work, and insert blockers above this point of plan into stack if issues arise. Basically, we need to asser this issue will be handled. Not necessarily it will happen right now. 

**Invariants this unit must plan to satisfy:**
- `ShramForCausalLM` overrides `_prepare_cache_for_generation` to instantiate
  `ShramCache` with parameters derived from model config.
- Calling `model.generate()` without an explicit `past_key_values` produces a
  `ShramCache` internally — the caller does not need to construct it.
- A caller who explicitly passes `past_key_values=ShramCache(...)` has it used
  unchanged — the supplied cache is not replaced.
- Tests verify: generate() without explicit cache produces ShramCache; generate()
  with explicit ShramCache uses it unchanged.

**Research notes (confirmed):**
- `_prepare_cache_for_generation(generation_config, model_kwargs, generation_mode,
  batch_size, max_cache_length)` is the override point. It writes to
  `model_kwargs["past_key_values"]` as a side effect.
- GenerationMixin computes cache_position as torch.arange(N, N+M) for a block of
  M new tokens when N tokens are already cached — confirmed via web research.
---

### Unit 6.D — MoSRAH Sequence Position Decoder: Specification and Blocker

**Responsibility:** Promote the semantic-sequence position computation from informal notes
in Unit 6.0 to a formal, verified specification. Register a blocker against Unit 10 (MoSRAH)
requiring this specification to be satisfied before MoSRAH implementation begins. No code
is written in this unit — the output is a plan entry with invariants that Unit 10 must
implement.

**Invariants this unit must satisfy:**
- The semantic-sequence mode position algorithm (vectorized cumsum over selected_heads) is
  written up as a formal specification. This may be a unit before unit 10, or part of unit 10.
- The main-sequence mode (position_ids passed through unchanged) is also formally specified
  in the Unit 10 entry for completeness.
- The position comes before the current unit 10 so it acts as a blocker. 

**Research notes (from Unit 6.0):**
Semantic-sequence mode algorithm:
1. Allocate a (B, N, L) boolean tensor, filled False.
2. Scatter True along the L dimension using `selected_heads` (B, N, K) — marking which
   heads each token routed to.
3. Cumsum along the N (sequence) dimension. Result[b, n, l] = number of times head l
   has been routed to by tokens 0..n in batch b.
4. Subtract 1 to get zero-based local position within the head's slot.
5. Add the expert lengths. Is zero when training, or get_expert_lengths during decoding. 
6. Gather along the L dimension using `selected_heads` to produce the final position
   tensor of shape (B, N, K).

---

### Unit 7 — Local Sliding-Window Attention Module

Sliding window attention is required for the functionality of the SHRAM system. We shall implement it using flex attention. 

**Responsibility:** Implements h_l(x) as a standalone verified module before the
hybrid layer (Unit 10) assembles it. Ensure RoPE is built in it such that it works from the absolute positional embeddings, and such that rope does not dilate when switching to long sequence mode. Interface with the cache correctly. During developement, halt and handle any blockers discovered.

**Invariants this unit must satisfy:**
- Implements h_l(x): (B, N, d) → (B, N, d).
- Local attention is enforced by the kernel natively — window size is passed to
  the kernel itself. A boolean attn_mask constructed outside the kernel is not
  acceptable (job.md Architecture §). OD-1 resolved: flex_attention with
  create_block_mask is used. BlockMask skips fully-masked blocks at zero FLOP cost.
  User has approved flex_attention as equivalent to FlashAttention for this purpose.
- Window size is configurable via config. Nothing is hardcoded.
- The module is causal within the window.
- h_l constructs its own `RotaryEmbeddings. Positions are passed in externally as an argument
- 
- h_l calls `local_cache.update(k, v, layer_idx)` directly on the sub-cache — not
  through ShramCache.update(). Returns the full window of accumulated K/V.
- MHA (not GQA) — OD-3 resolution.
- Tests verify: output shape; tokens outside the window receive zero attention
  weight; causal ordering preserved within window; window_size equal to sequence
  length degrades to standard causal attention.

**Paper refs:** Appendix A.Local Attention. **job.md ref:** Architecture § (local path).


---

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

**Paper refs:** Appendix A.Expert Packing and Unpacking.

---

### Unit 9 — Bottlenecked Ensemble Attention (BEA)

**Invariants this unit must satisfy:**
- BEA operates on `packed_hidden` (B, L, T, d); output y (B, L, T, d).
- W_Q, W_K, W_V have shape (L, d, `mosrah_head_dim`); W_O has shape (L, `mosrah_head_dim`,
  d). Each of L heads has independent parameters — no weight sharing across heads.
- BEA constructs its own `RotaryEmbedding(mode="yarn", head_dim=config.head_dim,
  theta=config.mosrah_rope_theta, initial_seq_length=config.training_sequence_length,
  dilation=config.scale, alpha=config.alpha, beta=config.beta)`.
  The returned `A_rope` is applied to attention logits before softmax.
- RoPE is applied with the supplied position tensor (either `packed_positions` for
  main-sequence mode or local slot indices for semantic-sequence mode). BEA passes the
  tensor; it does not choose the mode — that is the caller's (MoSRAH's) responsibility.
- Causal masking is a triangular mask over the T (packed) dimension.
- Padded positions (`active_mask` == False) do not contribute to attention output.
- Tests verify: output shape, head independence (perturbing one head's weights changes only
  that head's output), RoPE passthrough (both position tensors produce different outputs),
  padding positions do not influence real-token outputs, causal masking holds.

**Paper refs:** Appendix A.BEA, Appendix B.RoPE Mechanics.

---

### Unit 10 — MoSRAH Sparse Path

**Invariants this unit must satisfy:**
- MoSRAH forward: (`selected_heads`, `routing_probs`) = Router(x); `packed_hidden` =
  Pack(x, `selected_heads`); y = BEA(`packed_hidden`); `unpacked_output` = Unpack(y);
  o = sum_k `unpacked_output`_k * `routing_probs`_k. Matches Algorithm 1 in the paper.
- Weighted reduction uses `routing_probs` (unbiased, renormalized) — not biased scores,
  not `selected_heads` indices.
- load_balance_loss from Router is propagated through MoSRAH's return.
- Input/output shape: (B, N, d) → (B, N, d). MoSRAH presents the same interface as a
  standard attention operator to the SHRAM hybrid layer.
- Tests verify: output shape, gradient flows through `routing_probs` to router weights (not
  just through direct attention path), load_balance_loss is present, weighted reduction is
  correct.

**Paper refs:** §3 Design, Algorithm 1, Appendix A throughout.

---

### Unit 11 — SHRAM Hybrid Layer

**Invariants this unit must satisfy:**
- SHRAM forward: H(x) = h_l(x) + h_s(x), where h_l is the local sliding-window module
  (Unit 7, verified black box) and h_s is MoSRAH (Unit 10, verified black box).
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
- load_balance_loss is accessible from the forward output so the training loop can weight and
  apply it. The consumer is responsible for the weight (1e-2 in the paper; job.md Architecture
  § confirms this is the training loop's responsibility).
- KV cache uses DynamicCache with virtual layer indexing (OD-2 resolution): each SHRAM layer
  computes `virtual_layer = layer_idx * num_mosrah_heads + head_idx` and accesses the cache
  directly. Total virtual layers = num_hidden_layers * num_mosrah_heads. ShramForCausalLM
  creates a DynamicCache of this size and passes it down. use_cache=True is supported.
- If virtual layer indexing proves unworkable during implementation, push a custom cache unit
  as a blocker before continuing.
- Tests verify: load_balance_loss in output, weighted combination works end-to-end, KV cache
  accumulates correctly across decode steps, virtual layer mapping is correct.

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
