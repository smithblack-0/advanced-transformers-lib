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
- [X] Unit 7 — Local sliding-window attention module (h_l)
- [X] Unit 8 — Expert packing and unpacking: permutation machinery, padding, masks
- [X] Unit 9 — Bottlenecked Ensemble Attention (BEA): per-head attention on packed tensors
- [X] Unit 10.A (Blocker) - Expert packing and unpackign was implemented incorrectly for compatibility with inference, including in the paper. change pushed to paper. Packing revisions.
- [X] Unit 10.B — MoSRAH position computation: P tensor for both modes and all cache states
- [X] Unit 10.C — MoSRAH sparse path: routing → packing → position computation → BEA → unpacking → weighted reduction
- [X] Unit 11.A — Bug in MoSRAH, RoPe, or Attention Masking
- [X] Unit 11.B — SHRAM hybrid layer: assemble H(x) = h_l (Unit 7) + h_s (Unit 10.B)
- [X] Unit 12 — DecoderLayer: replace attention sublayer, propagate load_balance_loss
- [X] Unit 13 — ShramModel: aggregate load_balance_loss in output
- [X] Unit 14.A (Blocker) — MaxVio routing-imbalance metric: computation and end-to-end threading
- [X] Unit 14.B (Blocker) — We do in fact need to support masking.
- [X] Unit 14.C (Blocker) — LocalSlidingWindowLayerCache: local cache with update/retrieval contract for masked continuation
- [X] Unit 14.D (Blocker) — Local cache wiring: replace DynamicSlidingWindowLayer in ShramLayerCache/ShramCache, retire scalar seq-length
- [X] Unit 14.E (Blocker) — SlidingWindowAttention: consume new local cache contract, construct effective mask from returned state
- [X] Unit 14.F (Blocker) — Expert packing and unpacking masked continuation delta: pack outer_active_mask as a third tensor through expert-major transformation; return unpacking_mask and active_mask as distinct outputs
- [X] Unit 14.G (Blocker) — MoSRAH router masked continuation delta: exclude dead tokens from routing frequency statistics and normalization
- [X] Unit 14.H (Blocker) — MoSRAHLayer masked continuation delta: wire updated router and pack_experts contracts into MoSRAHLayer call sites
- [X] Unit 14.I (Blocker) — Mask plumbing through SHRAMHybridLayer, DecoderLayer, and ShramModel
- [X] Unit 15.A — Audit report on tied embeddings
- [X] Unit 15.B — Custom output class for huggingface
- [X] Unit 15.C (Blocker) — get sequence length cache support
- [X] Unit 15.D ShramForCausalLM: expose load_balance_loss and max_vio; KV cache resolution
- [X] Unit 16 — upload_to_hub
- [ ] Unit 17 — documentation
- [ ] Unit 18 — end-to-end tests
- [ ] Unit 19 — final audit

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

**Surface decisions.** Autonomous decisions are permitted but must be reported to the user for review. Uncertain decisions must be escalated before proceeding. Do not resolve ambiguity silently.

**Keep this plan current.** Update the status section and unit checklist continuously. The plan must reflect actual state at all times so work can be resumed after a session break without loss.

**Plan first within each unit.** At the start of each unit, state the invariants the unit must satisfy before considering any implementation. Implementation follows from invariants. A unit whose implementation is planned before its invariants are stated is not planned — it is guessed.

**Close the testing gap.** When a defect is found — in audit or anywhere else — the resolution is not complete until two questions are answered: (1) what invariant did the existing tests fail to enforce, and (2) what test change closes that gap? The fix and the test correction are a single
unit of work.

**Surface paper↔implementation conflicts.** Where the paper is silent, incomplete, or leaves a value undetermined, that is a decision point — not a license to fill the gap. Surface it, resolve it with the user, and record the resolution here before proceeding.

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
- The unit defines `SlidingWindowAttention` for one SHRAM decoder layer only.
- The unit implements the local short-range dot-product attention path inside SHRAM.
- The unit implements causal sliding-window attention.
- The unit accepts and uses a SlidingWindowLayerCache. This is retrieved by the parent layer off the `ShramLayerCache`.
- The unit supports cached execution for later experimentation.
- The unit implements positional encoding for the local path.
- The local path's positional encoding uses `default` RoPE mode and does not respond to YaRN
  dilation or long-context scaling changes elsewhere in the model.
- The unit returns outputs in `(B, N, d)` form compatible with downstream SHRAM composition.
- The unit is compatible with the broader SHRAM layer and cache architecture.
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
- The layer will receive and consume a DynamicSlidingWindowLayer huggingface cache.
- If FlexAttention continues to satisfy the required sliding-window, causal, and long-sequence
  behavior cleanly, it should be the preferred backend for this unit.
- The current RoPe design should be usuable in the default mode. If it is not it is a blocker and should be raised and discussed.
- If cached and uncached execution diverge in behavior, correctness takes priority and the
  divergence should be surfaced explicitly rather than hidden behind fallback logic.
- If the chosen backend introduces a limitation that breaks any of the above invariants, that
  limitation should be surfaced and the backend choice reconsidered explicitly rather than worked around silently.
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
- (Note on last case): Verify a deliberately unstable-sort alternative would fail the causal-order test.
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

---

### Unit 9 — Bottlenecked Ensemble Attention (BEA)

**Responsibility:** Define and verify the Bottlenecked Ensemble Attention operator used by the MoSRAH sparse path. This unit consumes packed expert-choice tensors together with a supplied position tensor and optional layer-local sparse cache, and returns expert-choice outputs in the same packed space.

**Context of Correctness**

BEA is the expert-choice attention operator in the MoSRAH system. It consumes packed expert tensors with some amount of padding that will later be removed during unpacking. It is essencial it operates causally with some sort of attention formulation. Importantly, BEA does not directly determine its rope position and must operate correctly in an inference mode. It will receive a cache for this purpose. The implementation is already highly constrained by the requirements of the paper.

**Invariants this unit must satisfy:**
- BEA is implemented as the packed expert-choice attention operator for the sparse MoSRAH path.
- The unit consumes packed expert-choice hidden states, a supplied position tensor, an
  active-token mask, and an optional layer-local MoSRAH cache.
- The unit does not compute RoPE positions but uses the provided one. 
- The unit instances its RoPE mode in the YaRN capable mode. 
- The unit implements interface
  `forward(packed_embeddings, position_ids, active_mask, cache=None)` where:
  - `packed_embeddings` has shape `(B, L, T, d)`
  - `position_ids` has shape `(B, L, T)`
  - `active_mask` has shape `(B, L, T)`
  - return value has shape `(B, L, T, d)`
- The BEA projection parameters are independent per head. No cross-head parameter sharing is introduced here. See the paper.
- BEA applies RoPE to `Q` and `K` using the supplied `position_ids` and consumes the returned attention scaling `A_rope` in the attention computation. It does so in the YaRN capable mode.
- If a cache is provided, BEA stores post-RoPE `K̃` and raw `V` into that layer-local sparse  cache and performs attention against the accumulated cached state returned by the cache.
- If no cache is provided, BEA performs attention against the current-step `K̃`, `V`, and
  `active_mask` directly.
- Causal masking is triangular over the packed sequence dimension or an equivalent mathematical system is utilized to support causality. Care is taken to maintain causality in inference mode as well.
- Inactive / padded positions do not contribute to SHRAM system outputs. This is not the same as not contributing to the BEA output. Largely this means we do not have to zero out junk queries, we just have to avoid attending to it. 
- The unit returns packed expert-choice outputs in the same `(B, L, T, d)` space expected by later unpacking.
- Tests and design for this unit are themselves structured so the unit can be trusted as a
  verified black box by later sparse-path assembly.

**Tests**
- Verify output shape is correct.
- Verify BEA accepts packed expert-choice inputs and returns packed expert-choice outputs.
- Verify per-head parameter independence.
- Verify different supplied position tensors produce different RoPE-adjusted attention behavior.
- Verify BEA does not internally compute positions or override the supplied RoPE mode when RoPE is activated in YaRN.
- Verify junk attention positions do not influence active-token outputs.
- Verify causal masking is preserved over packed positions.
- Verify cached execution stores post-RoPE `K̃` and raw `V`.
- Verify cached execution uses accumulated cached state returned by the provided local MoSRAH
  cache.
- Verify uncached execution uses the current `K̃`, `V`, and `active_mask` directly.
- Verify cached and uncached execution both satisfy the same externally visible BEA contract and produce identical results.
- Verify the implementation uses a non-naive attention path compatible with the tensor form.
- Verify the system responds to rescales under YaRN

**Preliminary implementation strategy:**
- Cached and uncached BEA should share one public operator contract, with a small branch on whether a local sparse cache was actually passed.
- It is almost assuredly the case that the same mask can be used in both RoPE modes, as both modes in essence just care about causal connections. This should be verified against the paper, however.
- The causal mask requirements will shift when operating in inference mode, as keys may only raggedly corrolate with queries. This will require generating masks based on the actual query positions.
- If the underlying kernel supports it, it would be a good idea to exclude dead query calls from being attended to to save flops. Naturally, dead keys should never be attended to, and dead queries would nonetheles be removed by unpacking, but this would safe flops.
- If a cache is passed, the operator should route through that cache using post-RoPE in the expert-packing mode. `K̃`, raw `V`, and the active-token mask; otherwise it should attend directly over the current-step  tensors.
- If the tensor forms remain ordinary headed attention tensors, a fast attention backend such as FlashAttention or FlexAttention should be preferred over a naive implementation.
- If any implementation choice would cause RoPE to be applied twice to cached keys, it should be  rejected.
- If a more efficient backend is introduced later, it must preserve the same externally visible  packed-space behavior, masking behavior, and cache semantics.
- If assumptions are not satisfied, they should instead be explicitly surfaced, handled, and documented as plan deviations.

---
### Unit 10.A — Main-Sequence Position Packing Fix

**Responsibility:** Define and verify the blocker fix required to make packed original-token
positions correct under both training and cached inference.

**Context of Correctness**

Originally, it was not caught that positional index tensors had to be packed too. This functionality needed to be added to the paper and needs to be added to the packing functions. While the paper has been corrected and is now again the authorative algorithmic source, the fix needs to be put in place. This change is needed as during inference RoPE needs to start at positions beyond just zero.

**Invariants this unit must satisfy:**
- This unit fixes the packing path so that original-token positions used for main-sequence RoPE
  are sourced from upstream position state rather than synthesized locally from the current chunk.
- The updated paper is the source of truth for the algorithmic requirement this unit implements.
- The packing path preserves upstream-provided original-token positions through the same
  expert-choice rearrangement used for routed token copies.
- Training/full-sequence execution remains valid when upstream position state is the ordinary
  sequential token positions.
- Cached inference remains valid when upstream position state is already advanced beyond the
  current local chunk.
- The resulting packed original-token positions remain aligned with the packed expert-choice
  hidden-state ordering.
- This fix does not change the externally visible expert-major ordering, active-mask semantics,
  padding semantics, or inverse mapping behavior of the packing/unpacking subsystem, except where
  necessary to correct the source of packed positions.
- This blocker exists specifically to make main-sequence position handling correct under cached
  inference; semantic-sequence position logic remains a separate concern.
- Tests and design for this unit are themselves structured so the corrected packing path can be
  trusted by later position and sparse-path units.

**Tests**
- Verify the packing path accepts authoritative upstream original-token position state.
- Verify training/full-sequence behavior remains correct when upstream position state is the
  ordinary sequential token positions.
- Verify cached inference preserves advanced upstream positions through packing rather than
  collapsing them to local chunk indices.
- Verify packed position outputs remain aligned with packed hidden-state ordering.
- Verify active-mask behavior is unchanged by this fix.
- Verify unpacking/inverse-mapping behavior is unchanged by this fix.
- Verify main-sequence cached inference no longer depends on locally regenerated chunk positions.

**Preliminary implementation strategy:**
- The updated paper should be treated as the primary algorithmic specification for this blocker.
- If upstream position state is already available at the caller boundary, packing should treat it
  as authoritative and only rearrange it.
- If any current implementation path regenerates original-token positions from local sequence
  length, that path should be removed or restricted to the special case where upstream position
  state is explicitly that same sequence.
- If later refactors change packing internals, they must continue to preserve authoritative
  upstream original-token positions exactly through the packing transformation.

### Unit 10.B — SparseMoSRAHPositions

**Responsibility:** Define and verify the `SparseMoSRAHPositions` layer used to compute the position tensor supplied to BEA for the MoSRAH sparse path.

**Context of Correctness**

A degree of ambiguity exists in what form of RoPE to use in this kind of model. Does the position as measured in the expert choice packing, or original token-choice packing matter more? The paper was unable to determine it so we must support both. In main-sequence mode, the packed original-token positions from Unit 8 are used directly. In semantic-sequence mode, positions are instead local to each expert bucket. This is a separate unit as it was deemed sufficiently complex it should be it's own module. The algorithm is highly constrained as research was already performed during the implementation of unit 6. The responsibility is implemented as a layer.

**Invariants this unit must satisfy:**
- This unit is implemented as a `SparseMoSRAHPositions` layer configured from config.
- This unit computes the position tensor `P` supplied to BEA for the MoSRAH sparse path in both inference and training configrations.
- There are two supported RoPE modes:
  - `main_sequence`
  - `semantic_sequence`
- In `main_sequence` mode, `P = J` always, where `J` is the packed original-token position
  tensor produced by Unit 8.
- In `semantic_sequence` mode with no sparse cache present, `P` is the per-expert local sequence
  `0, 1, 2, ...` over the packed sequence dimension when training, or the equivant shifted formulation n, n+1, n+2... in inference. 
- In `semantic_sequence` mode with a sparse cache present, `P` is that same local sequence
  offset by the current per-head occupancies returned by `get_heads_lengths()`.
- The cached and uncached semantic-sequence computations produce contiguous per-expert positions and are the same (equivalent in nature).
- The resulting `P` has shape `(B, L, T)` and is compatible with BEA.
- The layer receives the MoSRAH cache for inference mode computations.
- Tests and design for this unit are themselves structured so the unit can be trusted as a
  verified black box by Unit 10.B.

**Tests**
- Verify `main_sequence` mode returns `J` unchanged.
- Verify `semantic_sequence` mode without cache returns local per-expert positions
  `0, 1, 2, ...`.
- Verify `semantic_sequence` mode with cache returns occupancy-offset local positions using
  `get_heads_lengths()`.
- Verify bulk updates with `T > 1` produce contiguous positions.
- Verify cached semantic-sequence computation reads `get_heads_lengths()` before any cache update
  in the same forward pass.
- Verify returned position tensors have the correct shape for BEA.
- Verify the configured layer selects the correct computation for each supported RoPE mode.

**Preliminary implementation strategy:**
- If `semantic_sequence` mode is selected and no contradictory ground truth is discovered during
  implementation, this layer should first generate the local `arange` positions, then add cache
  offsets if a sparse cache is present.
- If `main_sequence` mode is selected, the layer should forward the packed position tensor `J`
  directly.
- If any later implementation change would cause BEA to update the cache before this layer reads
  cached occupancies, that change should be rejected.
- If the layer can cleanly encapsulate all RoPE-position branching without duplicating logic in
  the parent MoSRAH path, that should be preferred.

### Unit 10.C — MoSRAH Path

**Responsibility:** Define and verify the full MoSRAH sparse path from routed token-choice input to final model-space sparse output.

**Context of Correctness**

The SHRAM system requires packing into an expert choice form to surface the long-sequence guarentees. The MosRAH system has to put the pieces together. Routing funtions produced in the routing unit, and the BEA layer, have to work together to attend correctly. Additionally, it is essential that position be correctly encoded for the two modes of rope MoSRAH supports. It must support inference through the layer-local MoSRAH cache, and generate the needed load-balancing signal. Ultimately, this is largely a coordinator and constructor.

**Invariants this unit must satisfy:**
- The MoSRAH path is implemented as specified by Algorithm 1 and the surrounding appendix
  material in the paper.
- The unit consumes model-space hidden states `x` of shape `(B, N, d)` and returns model-space
  sparse outputs of shape `(B, N, d)` together with `load_balance_loss`.
- The unit performs the sparse-path forward sequence in the following order:
  1. routing
  2. expert packing
  3. position computation
  4. BEA
  5. expert unpacking
  6. weighted reduction
- Routing produces `(selected_heads, routing_probs)` in token-choice form.
- Packing consumes `x` and `selected_heads` and produces the packed expert-choice state required
  by BEA.
- Position computation is delegated to Unit 10.A.
- BEA is called on packed expert-choice tensors and receives the layer-local MoSRAH cache
  directly if caching is enabled.
- Unpacking restores token-choice ordering from BEA's packed expert-choice outputs.
- The final weighted reduction uses the unbiased renormalized `routing_probs`.
- The unit supports both RoPE position modes provided by Unit 10.B.
- The unit supports both cached and uncached execution.
- `load_balance_loss` from the router is propagated through the MoSRAH path return.
- Underlying features and code are utilized to accomplish this without reinventing the wheel.
- Tests and design for this unit are themselves structured so the unit can be trusted as a
  verified black box by later SHRAM-layer assembly.

**Tests**

*Note: As an orchestrator, what is required is that between this and it's subunits all parts are tested*

- Verify output shape is correct.
- Verify `load_balance_loss` is present in the unit return.
- Verify the final weighted reduction uses `routing_probs`.
- Verify the final weighted reduction does not use biased routing scores.
- Verify manually configured weights and manually specified inputs operate as expected based on the paper. 
- Verify gradients flow through the loss and through the SHRAM layer. 
- Verify uncached execution works correctly.
- Verify cached execution uses the provided layer-local MoSRAH cache and accumulates correctly
- Verify uncached and cached execution arrive at the same result.
  across calls.
- Verify cached and uncached execution both satisfy the same external MoSRAH-path contract.

**Preliminary implementation strategy:**
- The paper should be treated as the primary algorithmic specification for this unit. 
- If no contradictory ground truth is discovered during implementation, this unit should be written as a straightforward orchestration of the already-verified subunits rather than as a fused monolith that hides the bridge steps.
- If caching is enabled, the unit should pass the layer-local MoSRAH cache directly into BEA;  otherwise it should execute the uncached path with the current packed tensors and active mask.

### Unit 11.A — Bug in RoPE or MoSRAH

**Symptoms**

Despite thorough verification of test setup, a bug persisted in which switching between a `main_sequence` RoPE configuration and a `semantic_sequence` configuration had no effect on the outcome, violating the test condition.

**Debugging**

The problem was tracked down to the flex attention and masking unit. Debugger inspection confirmed that RoPE was functioning correctly: queries and keys were being rotated differently under the two modes, and position IDs differed between them as expected. This made the subsequent observation deeply misleading — the attended states were identical despite the inputs being visibly different. This pointed squarely at a fault in the attention or masking logic rather than in RoPE, directing investigation toward the attention system on that basis.

**Resolution**

This turned out to be a horribly unlucky coincidence. RoPE encodes position as a phase relationship between pairs of tokens — what matters is not the absolute position of any token, but the difference between the query's position and the key's position. While the position IDs were different between modes, the active (unpadded) regions in both `semantic_sequence` and `main_sequence` modes happened to consist entirely of contiguous integer sequences. Any contiguous sequence produces the same pairwise differences regardless of where it starts, so both modes produced identical attention patterns despite carrying visibly different position tensors.

What looked like a fault in the attention system was in reality a test design flaw: the structured input and the particular weight initialization colluded to produce contiguous position assignments in both modes, making them indistinguishable to the attention mechanism. The test has been fixed and the blocker cleared.

This was an exceptionally unpleasant bug to work through, and honestly one of the worst I can remember — the evidence was internally consistent, the logic was sound at every step, and the actual problem was not where any of it pointed.

### Unit 11.B — SHRAM Hybrid Layer

**Responsibility:** Define and verify the SHRAM hybrid attention layer that combines the local
sliding-window path and the MoSRAH sparse path at one decoder attention slot.

**Context of Correctness**

SHRAM is the hybrid attention construction used by the model. The local path exists to preserve
nearby-token behavior, while the MoSRAH sparse path is the theorem-facing long-range path. Both
live at the same decoder attention slot and both must operate over the same input hidden state.
The layer-local cache architecture already reflects this: each `ShramLayerCache` owns both a
`.sliding_window_cache` and a `.mosrah_cache`. The hybrid layer therefore exists to dispatch the
correct cache to each path, execute both paths, sum their outputs, and propagate the sparse-path
load-balance signal.

**Invariants this unit must satisfy:**
- The SHRAM hybrid layer implements the forward relation `H(x) = h_l(x) + h_s(x)`.
- `h_l` is the verified local sliding-window attention path from Unit 7.
- `h_s` is the verified MoSRAH sparse path from Unit 10.C.
- Both paths consume the same input hidden state for the current decoder layer.
- The hybrid layer receives the corresponding `ShramLayerCache` for this decoder layer.
- The hybrid layer passes `DynamicSlidingWindowLayer` to the local path.
- The hybrid layer passes `MoSRAHCache` to the MoSRAH sparse path.
- `h_l` and `h_s` have fully independent parameters.
- The hybrid output is the sum of the two path outputs in model space.
- `load_balance_loss` from the sparse path is propagated through the hybrid layer's return.
- The hybrid layer returns outputs in `(B, N, d)` form compatible with the decoder-layer
  interface it replaces.
- Tests and design for this unit are themselves structured so the unit can be trusted as a
  verified black box by the decoder-layer unit.

**Tests**

*Note: As an orchestrator, what is required is that between this and it's subunits all parts are tested*

- Verify output shape is `(B, N, d)`.
- Verify zeroing or disabling the local path leaves only the sparse-path contribution.
- Verify zeroing or disabling the sparse path leaves only the local-path contribution.
- Verify `load_balance_loss` is present in the return and is propagated from the sparse path.
- Verify the system responds to the load balancing loss when gradients occur. Verify movement occurs in the corret direction.
- Verify the hybrid layer preserves the external attention-layer interface expected by the
  decoder layer.

**Preliminary implementation strategy:**
- This unit should be a straightforward coordinator over the already-verified local and sparse sublayers rather than a from-scratch implementation.
- If the local and sparse paths already return model-space tensors of matching shape, the hybrid
  layer should sum them directly rather than introducing any additionallogic here.
- If a `ShramLayerCache` is provided, this unit should dispatch its two owned sub-caches to the
  corresponding paths rather than re-resolving cache ownership internally.
- If later performance work attempts to fuse the two paths more aggressively, it must preserve
  the same externally visible sum, cache dispatch behavior, and load-balance-loss propagation.
---

### Unit 12 — DecoderLayer Update

**Invariants this unit must satisfy:**
- DecoderLayer uses SHRAM hybrid layer in place of GroupedQueryAttention.
- load_balance_loss returned by the SHRAM layer is propagated through DecoderLayer's output.
- The pre-norm + residual structure is unchanged: x' = x + SHRAM(RMSNorm(x));
  x'' = x' + FFN(RMSNorm(x')). This structure transfers unchanged per Appendix A.
- Tests verify: output shape unchanged, load_balance_loss present in output, structure holds.


### Unit 13 — ShramModel Update

**Responsibility:** Define and verify the backbone-level model boundary above `DecoderLayer`.
This unit owns stack composition of decoder layers, final normalization, model-level
`load_balance_loss` aggregation, model-level cache routing through `ShramCache`, and preservation
of the existing backbone output-dict semantics.

**Context of Correctness**

`ShramModel` is a simple orchestration unit, not a new semantics unit. It should not re-prove
decoder, SHRAM, BEA, or cache-internal correctness already owned by lower units. Its job is to
assemble already-verified decoder layers into the backbone contract the rest of the model depends
on.

Because this unit is small and structurally transparent, some facts are better verified by audit
than by brittle internal tests. Runtime tests should focus on public behavior. Structural wiring
facts that are obvious by direct inspection and would require implementation-shaped tests to
enforce should instead be recorded here as audit requirements and checked independently during
verification.

**Invariants this unit must satisfy:**
- `ShramModel` iterates the decoder stack and applies the final RMSNorm.
- `ShramModel` collects `load_balance_loss` from every decoder layer and exposes a single scalar
  `load_balance_loss` in the output dict.
- Model-level `load_balance_loss` is the **sum** of the per-layer decoder losses.
- The existing output dict keys are preserved:
  - `last_hidden_state`
  - `past_key_values`
  - `hidden_states`
- `last_hidden_state` preserves its existing semantic role as the final normed backbone output.
- `past_key_values` preserves its existing semantic role as the model-level cache object returned
  from the backbone boundary.
- `hidden_states` preserves its existing semantic role when `output_hidden_states=True`, and is
  `None` when `output_hidden_states=False`.
- The backbone continues to accept pre-embedded `inputs_embeds`, not token IDs.
- The model-level cache boundary is `ShramCache`; per-layer routing is through `cache.layers[i]`.

**Tests**
- Verify the real forward pass returns a plain dict with the expected keys, including
  `load_balance_loss`.
- Verify `last_hidden_state` has the correct shape and `load_balance_loss` is a finite scalar.
- Verify `hidden_states is None` when `output_hidden_states=False`.
- Verify `hidden_states` has length `num_hidden_layers + 1` when `output_hidden_states=True`.
- Verify `past_key_values is None` when no cache is provided.
- Verify the same top-level `ShramCache` object is returned when one is provided.

**Audit requirements**
- Inspect `ShramModel.forward` and verify that every decoder layer is called exactly once, in
  order, over the evolving hidden-state stream.
- Inspect `ShramModel.forward` and verify that per-layer `load_balance_loss` values are summed
  across all decoder layers rather than overwritten, ignored, averaged, or taken only from the
  final layer.
- Inspect `ShramModel.forward` and verify that `cache.layers[i]` is passed to decoder layer `i`.
- Inspect `ShramModel.forward` and verify that `hidden_states` are collected with the preserved
  backbone semantics rather than accidentally collecting tuples or post-final-norm values.
- Inspect `ShramModel.forward` and verify that the final RMSNorm is applied exactly once at the
  model boundary to produce `last_hidden_state`.

**Preliminary implementation strategy:**
- Treat this as a backbone-boundary update, not a semantic redesign.
- Prefer real runtime smoke tests for public behavior and explicit audit for obvious wiring facts.
- Do not introduce fake decoder layers, handwritten alternate forward passes, or re-proofs of
  lower-level attention/cache semantics in this unit’s tests.

### Unit 14.A (Blocker) — MaxVio Routing-Imbalance Metric

**Responsibility:** Compute the MaxVio routing-imbalance scalar in the router and thread it
through the MoSRAH sparse path, SHRAM hybrid layer, decoder layer, and ShramModel so that it
is available at the ShramForCausalLM boundary for Unit 14.B to expose.

**Context of Correctness**

MaxVio is a scalar summary of routing imbalance defined in the paper (§MaxVio):

    MaxVio = L · max_l(f_l − 1/L)

where f_l is the realized routing frequency of head l and 1/L is the perfectly balanced target.
A value of zero indicates perfect balance.

This metric was not included when the router, MoSRAH sparse path, hybrid layer, decoder layer,
and ShramModel were originally implemented and verified (Units 4, 10.C, 11.B, 12, 13). Those
units are complete and their audit records stand. This blocker surfaces as a new requirement at
the boundary of Unit 14.B, which must expose MaxVio in the model output. Because those earlier
units are append-only audit records, the threading work is a new forward step rather than a
revision of prior work.

The router already computes `routing_freqs` (f_l) internally as part of the load-balance loss
machinery. MaxVio is a one-line derivation from that quantity and belongs in the router, which
is the only component with direct access to per-head assignment counts. No other component in
the threading path should recompute or reinterpret it — they pass it through unchanged. MaxVio
is a monitoring scalar, not a loss; it must be detached from the autograd graph at the point of
computation and must never contribute gradients to any parameter.

**Invariants this unit must satisfy:**

- The router computes MaxVio as `L · (routing_freqs − 1/L).max()` from the already-computed
  `routing_freqs` and returns it as an additional output alongside `selected_heads`,
  `routing_probs`, and `load_balance_loss`.
- MaxVio is detached from the autograd graph at the point of computation in the router. No
  downstream component re-attaches it.
- MaxVio flows from the router through MoSRAH, through the SHRAM hybrid layer, through the
  decoder layer, and through ShramModel without modification or reinterpretation.
- ShramModel aggregates per-layer MaxVio values as the maximum across all decoder layers. The
  maximum is the correct aggregation because the anomaly recovery protocol responds to the
  worst-case head imbalance across the model, not an average.
- The threading additions to MoSRAH, SHRAMHybridLayer, DecoderLayer, and ShramModel do not
  alter any existing return values, invariants, or behavior verified in prior units.
- Tests and design for this unit are structured so that the MaxVio threading can be trusted as
  a verified contract by Unit 14.B.

**Tests:**

- Verify MaxVio is exactly 0 when all heads receive equal routing frequency.
- Verify MaxVio is exactly 1 when the most overloaded head receives double its fair share.
- Verify MaxVio produces the correct value for a known intermediate routing imbalance.
- Verify MaxVio is detached (`requires_grad` is False and it does not appear in any parameter's
  gradient graph).
- Verify MaxVio propagates through the SHRAM hybrid layer and is present in the decoder layer
  output.
- Any existing output-existence tests in the units this value threads through (MoSRAH,
  SHRAMHybridLayer, DecoderLayer, ShramModel) must be updated to also verify the presence of
  `max_vio` in their return values.

**Audits**

- Verify the threading additions do not alter `load_balance_loss`, hidden state outputs, or any
  other previously verified return values.
- Verify ShramModel's output contains a MaxVio scalar equal to the maximum across all decoder
  layers.

**Preliminary implementation strategy:**

- Extract the MaxVio formula into a private helper (e.g. `_compute_max_vio(routing_freqs,
  num_heads)`) on the router. The router calls it after computing `routing_freqs`; tests call
  it directly with synthetic inputs, bypassing TopK entirely. This avoids the tie-breaking
  ambiguity that arises when all routing scores are equal.
- Threading through MoSRAH, SHRAMHybridLayer, and DecoderLayer is purely additive — receive,
  pass through. No logic change is needed at any intermediate layer. A minimal amount of
  adaptation is needed at each signature boundary.
- ShramModel collects one MaxVio per decoder layer and reduces with `.max()` before including
  it in the output. This finds the maximum imbalance and aligns with the existing max system at
  the frequency level.
- If any existing tests break due to the changed return signature of an intermediate component,
  those tests must be updated to reflect the new correct signature before this blocker is
  considered verified. If how to update them is not clear, a team decision is needed.

### Unit 14.B (Blocker) — Masked Continuation Support for SHRAM Attention Paths

**Responsibility:** Define and verify masked continuation support for SHRAM so that active/dead token information carried during continuation is preserved truthfully through the local sliding-window path, the sparse MoSRAH path, and the cache boundaries they depend on.

**Context of Correctness**

The existing SHRAM attention and cache boundaries were built for the regime where every token position in the active continuation state is semantically live. That is not sufficient once continuation may carry active/dead token information that must remain meaningful across the attention stack. The problem is not local to a single attention operator and it is not correctly solved at the later HuggingFace wrapper boundary. It is a cross-cutting attention-state issue that must be resolved inside the SHRAM cache and attention system before later wrapper work can be done truthfully.

This blocker therefore exists to establish a correct internal SHRAM contract for masked continuation. Later units may then rely on that contract as a verified black box rather than attempting to reconstruct masking semantics indirectly.

**Invariants this unit must satisfy:**

- SHRAM gains a truthful internal continuation-mask contract.
- Active token information is passed into the SHRAM stack at the `model.py` level as a mask of shape `(batch, current_sequences)`, where `True` means active. Notably, `current_sequences` is not always `total_sequences`: when generation is resuming, only the mask for the queries currently being attended to is passed forward.
- The local sliding-window cache preserves chronological cached local state by providing aligned key, value, and mask tensors, where the returned mask indicates whether each returned local token position is active (`True`) or dead (`False`).
- Ragged batch lengths are supported at the local-path boundary.
- While only a sliding window must be retained for the next step, the key/value/mask tensors returned for the current local-attention step must include all positions needed for correct local attention on that step.
- The local sliding-window attention path consumes the returned local key, value, and mask tensors and constructs its effective local causal/window visibility from them correctly.
- The MoSRAH cache pathway now properly packs away and uses the passed in masking entries.
- The cache system is updated to initialize and use the new local cache.
- The same current-chunk active mask is received by both the local path and the sparse path.
- Dead outer tokens do not become semantically active sparse-path token copies.
- Existing all-live behavior remains semantically equivalent to pre-blocker behavior.
- The broader SHRAM orchestration layers remain orchestration layers. They carry the new masking information without redefining attention semantics themselves.
- The resulting internal SHRAM contract is sufficient for later wrapper-level continuation support to be built against it truthfully.
- Tests and design are structured so this blocker can be trusted as a verified black box by the later `ShramForCausalLM` unit.

**Tests**

These are overall objectives.

- Verify local cached continuation preserves active/dead token state correctly across updates.
- Verify dead cached local tokens do not influence live local outputs.
- Verify sparse-path materialization excludes or suppresses dead outer tokens correctly.
- Verify packing and unpacking remain mutually correct under the new mask-aware sparse contract.
- Verify the same continuation mask reaches both local and sparse paths.
- Verify reset, reorder, and other supported cache permutations preserve the alignment of cached state and mask state.
- Verify all-live execution remains equivalent to the old behavior.
- Verify unsupported operations fail explicitly rather than returning misleading results.

**Preliminary implementation strategy:**

- Introduce a new local sliding-window cache variant whose contract is to accept local K, local V, and a chunk-local active mask, and to return chronological local K/V state together with an aligned mask to the local attention path.
- Keep the local cache logic simple: concatenate, trim/retain, and return aligned mask information; do not force the cache to fully solve ragged local visibility on its own. Accept that in ragged batches this may mean some returned local positions are dead. This is an acceptable trade because the MoSRAH path still handles the long-range dependencies.
- Update the local sliding-window attention path so it constructs its effective local causal/window mask from the returned local key/value/mask state.
- Thread current-chunk token masking from `model.py` through the SHRAM orchestration stack to both the local and sparse paths. Separately wire the special returned local cache.
- Apply outer token masking at the sparse-path materialization boundary, specifically where routed token copies are actually packed and later unpacked.
- Preserve the existing routing and other lower-level sparse mechanisms unless implementation proves that their semantics also need to become mask-aware. If this happens, discuss the issue.
- Keep the later HuggingFace wrapper rebuild out of scope for this blocker. That unit should consume the completed internal SHRAM contract rather than solve the masking problem itself.
- The MoSRAH system needs a little bit of modifications to packing to make sure it continues to work. 

### Unit 14.C — LocalSlidingWindowLayerCache

**Responsibility:** Define and verify `LocalSlidingWindowLayerCache`, the local-path cache for one SHRAM decoder layer. This unit owns the local cache contract for masked continuation and is responsible for accepting chunk-local key, value, and active-mask inputs, returning the current-step local frame consumed by local attention, and separately retaining the next-step sliding-window cache state.

**Context of Correctness**

Under masked continuation, the local path no longer operates in a regime where every stored local token position is semantically active. The masked-continuation contract now distinguishes between the local frame needed for the current attention step and the retained cache state needed for the next step. The local attention layer consumes the returned current-step frame and constructs effective causal/window visibility from the returned mask; the cache separately remembers the trimmed sliding-window state for later use. Downstream processing in the local layer already enforces a sliding window kernel, ensuring all we need to do is retain the context that will be needed on the next generation cycle.

**Invariants this unit must satisfy:**

- A new `LocalSlidingWindowLayerCache` is introduced as the local-path cache for one SHRAM decoder layer.
- `LocalSlidingWindowLayerCache` accepts local key, value, and active-mask inputs for the currently processed chunk.
- `LocalSlidingWindowLayerCache.update(keys, values, active_mask) -> (keys, values, active_mask)` is the authoritative update/retrieval interface for the local cache boundary.
- The returned key, value, and active-mask tensors are aligned in sequence dimension and are the concatenation of the internal mask and the input values. 
- The returned active mask indicates whether each returned local token position is active (`True`) or dead (`False`); downstream processors may worry about whether attention is justifed and what to do with the mask
- `LocalSlidingWindowLayerCache` separately trims and retains the next-step local cache state to the sliding-window buffer. This is used in the next step.
- `LocalSlidingWindowLayerCache` does not expose a single truthful scalar sequence length and raises `NotImplementedError` when asked for one.

**Tests**

- Verify `LocalSlidingWindowLayerCache.update(keys, values, active_mask)` returns aligned key/value/mask tensors of length sliding_window + sequence_length
- Verify repeated updates preserve chronological ordering under concatenation.
- Verify the retained next-step cache state is trimmed to the sliding-window buffer; this must be done by seeing if only that comes up in the next returned sequence call, not by direct inspection.
- Verify reset, reorder, repeat, and select preserve alignment of returned key/value/mask state and retained cache state.
- Verify `LocalSlidingWindowLayerCache.get_seq_length()` raises `NotImplementedError`.

**Preliminary notes:**
- Prefer a new cache file such as `sliding_window_cache.py`.
- Keep the cache contract simple: initialize a fixed sliding-window buffer, concatenate prior retained cache state and current chunk, return that concatenated current-step frame, and separately trim the retained next-step cache state to the sliding-window buffer.
- Treat the returned active mask as authoritative; do not require the cache itself to eliminate all dead positions from either the returned current-step frame or the retained next-step cache state.
- Preserve chronological ordering in both the returned current-step local frame and the retained next-step cache state.
- Avoid making the cache solve local causal masking by itself; that belongs to the later local-attention unit.


### Unit 14.D (Blocker) — Local Cache Wiring Through ShramLayerCache / ShramCache

**Change delta from 14.C:** Replace the old local cache type in the SHRAM cache stack and update the cache-boundary contracts accordingly.

**New / changed contract:**
- `ShramLayerCache` owns `LocalSlidingWindowLayerCache` as its local sub-cache in place of `DynamicSlidingWindowLayer`.
- `ShramCache` constructs layer caches using the new local sub-cache type.
- `ShramLayerCache` continues to expose the local and sparse sub-caches directly rather than introducing a composite update interface.
- `ShramLayerCache.get_seq_length()` raises `NotImplementedError`.
- `ShramCache.get_seq_length()` likewise raises `NotImplementedError`.
- Reset, reorder, repeat, select, offload, and prefetch continue to coordinate both sub-caches atomically across the cache stack.

**What should stay unchanged:**
- The SHRAM two-sub-cache architecture remains intact.
- Neither `ShramLayerCache` nor `ShramCache` gains a composite update interface.
- `ShramCache` remains the model-wide owner and coordinator, not a routing or masking unit.

**Tests delta:**
- Add coverage that `ShramLayerCache` owns `LocalSlidingWindowLayerCache` as its local sub-cache.
- Add coverage that `ShramCache` constructs the updated per-layer cache stack.
- Add coverage that `ShramLayerCache.get_seq_length()` raises `NotImplementedError`.
- Add coverage that `ShramCache.get_seq_length()` raises `NotImplementedError`.
- Update any existing tests that previously asserted scalar sequence length from the SHRAM cache stack.
- Add coverage that reset/reorder/repeat/select still coordinate the updated cache stack correctly.

**Preliminary notes:**
- Keep this unit narrow: it is ownership/wiring and cache-boundary contract adjustment, not local attention semantics.
- Update docstrings and cache-boundary comments so they no longer describe the old scalar sequence-length story or the old local cache type.

### Unit 14.E (Blocker) — SlidingWindowAttention Masked Continuation Delta

**Change delta from 14.B / 14.C / 14.D:** Update `SlidingWindowAttention` so the local path consumes the new local cache contract and derives local visibility from returned mask information rather than from raw returned buffer positions alone.

**Context of Correctness** 

LocalSlidingWindowLayerCache returns what is in essence the stored sliding window context concatenated with our new updates. It does no further processing than this. In order for the system to continue to operate correctly it is necessary to modify the local sliding window system's masking mechanism 14to both respond to a directly passed (no cache) mask and the mask that may be passed back by the sliding window cache. This is despite the fact packing may not be contingous; in theory, the cache can return a scenario where tokens 1, 2, 3 are live, 4, 5 are dead, and 6-10 are live again. Worse, this masking is different for different batches. The existing local path already owns local causal/window semantics and already uses a FlexAttention-style formulation. The modifications should be able to be performed by setting up the SlidingWindowAttention to accept the passed-in model.py mask format, run it throguh (or skip) the cache, then correctly construct and handle attention causally and locally regardless of what the mask ends up becoming.

**New / changed contract:**
- `SlidingWindowAttention` accepts the current-chunk active mask as described in 14.B in addition to its existing inputs.
- In the cache path, `SlidingWindowAttention` passes that mask into `LocalSlidingWindowLayerCache` and consumes the returned local key/value/mask frame.
- In the no-cache path, `SlidingWindowAttention` consumes the directly passed current-chunk mask.
- `SlidingWindowAttention` no longer treats raw returned buffer indices as sufficient to determine semantic local order.
- `SlidingWindowAttention` derives effective local visibility from the returned active-mask information; the fact that packing may not be left justified is properly handled by masking and flex attention.

**What should stay unchanged:**
- This remains the local short-range attention path for one SHRAM layer.
- It still owns local-path RoPE usage and local sliding-window semantics.
- This unit does not redefine broader SHRAM orchestration or sparse-path behavior.

**Tests delta:**
- Add coverage taht the no-cache and cache path continue to run correctly if not yet done.
- Add coverage verifying a cache path with a sequence of ragged batches constructs the correct expected masks.
- Add coverage physically verifying by a few hardcoded examples and contrast junk tokens have no effect on attention, while active tokens do. 
- Keep all-live equivalence coverage against the old local-attention behavior up to date.

**Preliminary notes:**
- Continue to use the FlexAttention-based local formulation.
- It is highly recommended to isolate the more complex masking functionality into a helper method for testing if not already done so. 
- Treat the returned local key/value/mask tensors as the authoritative local frame for this step.
- A good starting direction is to recover semantic active-token positions from the returned active mask, for example with a cumulative count over that mask.
- Use those recovered semantic positions, rather than raw returned buffer indices alone, to define causal precedence and sliding-window distance.
- Preserve the all-live path as a special case of the same formulation rather than introducing divergent masked and unmasked semantics.


### Unit 14.F (Blocker) — Expert Packing / Unpacking Masked Continuation Delta

**Context of Correctness**

The sparse path already has a masking subsystem at the BEA level and already relies on packing metadata to preserve causal order and to support unpacking. Under masked continuation, however, there is now a second masking signal: the current-chunk outer active mask passed down from `model.py`. That mask is not the same thing as the existing packed-slot occupancy mask and it does not enter the sparse path at the same boundary.

The packing boundary is therefore where these facts first meet. This is the point where token-choice routed data is transformed into expert-choice layout, where packed positions are already being constructed, and where the sparse path can still preserve the distinction between “this packed slot exists” and “this packed token is semantically active.” The BEA unit should continue to consume a packed mask; the work here is to make sure the correct mask is packed and made available.

**New / changed contract:**
- `setup_packing(...)` remains unchanged unless implementation proves otherwise.
- `pack_experts(...)` accepts the current-chunk outer active mask in addition to its existing inputs.
- `pack_experts(...)` packs that outer active-mask information through the same expert-choice transformation used for the routed token data.
- `pack_experts(...)` returns both:
  - the metadata needed for correct unpacking/restoration, and
  - the packed active-token mask needed by the existing sparse attention masking system.
- Dead outer tokens do not become semantically active packed sparse entries.
- The packed active-token mask remains aligned with the packed hidden states and packed positions.
- `unpack_experts(...)` consumes the updated packing contract and continues to restore token-choice ordering correctly.
- All-live execution remains equivalent to pre-blocker packing/unpacking behavior.

**What should stay unchanged:**
- The stable-sort expert-major ordering contract remains intact.
- The sparse attention unit continues to consume a packed mask rather than being redesigned around a new masking interface.
- This unit does not, by default, redefine router semantics. If implementation proves routing statistics must also become mask-aware, that is a surfaced issue rather than a silent change.

**Tests delta:**
- Add or verify coverage that dead outer tokens do not materialize active packed sparse entries.
- Add or verify coverage that the packed active-token mask is aligned with packed hidden states and packed positions.
- Add or verify coverage that packed positions remain aligned for live routed copies.
- Add or verify coverage that unpacking remains correct under the updated packing contract.
- Add or verify all-live equivalence coverage against the old sparse packing/unpacking behavior.

**Preliminary notes:**
- Keep this unit tightly constrained to the packing/unpacking boundary.
- Reuse the existing sparse attention masking subsystem rather than inventing a second one downstream.
- The preferred implementation direction is to pack the outer active mask through the same stable-sort / expert-choice transformation already applied to hidden states and positions. this should be another packing case like packing the position ids or the embeddings.
- A dictionary-style or auxiliary-structure return from `pack_experts(...)` is appropriate if it cleanly separates unpack/restoration metadata from the packed active-token mask. It may be worth a minor refactoring of unpack_experts to support this. 
- Do not modify BEA unless implementation proves that the existing packed-mask interface is insufficient. If this is encountered, raise the point for discussion

### Unit 14.G (Blocker) — MoSRAH Router Masked Continuation Delta

**Change delta from 14.B / 14.F:** Update `MoSRAHRouter` so routing statistics respond to the current-chunk outer active mask rather than treating dead tokens as valid contributors.

**Context of Correctness**

The sparse-path packing boundary now handles semantic suppression of dead outer tokens when routed token copies are materialized. However, the router computes routing statistics earlier, from the realized assignment tensor. This means dead outer tokens can still distort `load_balance_loss` and `max_vio` unless the router itself becomes aware of the current-chunk active mask. The router does not need a new routing algorithm. It only needs to ensure that dead tokens do not contribute to realized assignment frequencies or to the normalization used to derive those frequencies.

**New / changed contract:**
- `MoSRAHRouter` accepts the current-chunk outer active mask in addition to its existing inputs.
- `MoSRAHRouter` continues to compute `selected_heads` and `routing_probs` over the current chunk.
- The realized routing assignment tensor used for routing-frequency statistics is masked by the current-chunk outer active mask before reduction.
- Routing frequencies are normalized by the number of active token/head assignments rather than by the total raw token count.
- Dead outer tokens do not contribute to `load_balance_loss`.
- Dead outer tokens do not contribute to `max_vio`.
- All-live execution remains equivalent to pre-blocker router behavior.

**What should stay unchanged:**
- The router still performs the same Top-K token-choice routing over the current chunk.
- This unit does not redefine sparse-path packing or BEA masking.
- This unit does not, by default, change the meaning of `selected_heads` or `routing_probs` beyond excluding dead-token contribution from routing statistics.

**Tests delta:**
- Add coverage that dead outer tokens do not affect `load_balance_loss`.
- Add coverage that dead outer tokens do not affect `max_vio`.
- Add coverage that all-live execution remains equivalent to the old router behavior.
- Add coverage for the all-dead edge case if that case is allowed to occur.

**Preliminary notes:**
- Keep this unit tightly constrained to routing statistics.
- The preferred implementation direction is to apply the outer active mask to the realized assignment tensor after scatter and before frequency reduction.
- Normalize routing frequencies by the number of active token/head assignments rather than by the total raw token count.
- If the all-dead case is possible, define its router outputs explicitly rather than allowing divide-by-zero behavior.

### Unit 14.H (Blocker) — MoSRAHLayer Masked Continuation Delta

**Change delta from 14.F and 14.G:** Wire the updated contracts from `pack_experts` (14.F) and `MoSRAHRouter` (14.G) into `MoSRAHLayer` so the layer correctly calls its sub-components with the active mask they now require.

**Context of Correctness**

14.F and 14.G updated `pack_experts` and `MoSRAHRouter` to accept and use `outer_active_mask`/`active_mask`, but `MoSRAHLayer` still calls both with the old signatures. Three call sites are broken: the router call is missing `active_mask`, `pack_experts` is missing `outer_active_mask` and its 3-tuple unpack is now wrong (it returns 4), and `unpack_experts` receives `active_mask` where it now expects `unpacking_mask`. This unit fixes those call sites. It is internal rewiring of `MoSRAHLayer` — not plumbing above it, and not changes to the sub-components themselves.

**New / changed contract:**
- `MoSRAHLayer.forward` accepts `active_mask: torch.Tensor` of shape `(B, N)` in addition to its existing inputs.
- `MoSRAHLayer` passes `active_mask` to `MoSRAHRouter`.
- `MoSRAHLayer` passes `active_mask` as `outer_active_mask` to `pack_experts`.
- `MoSRAHLayer` correctly unpacks all four return values from `pack_experts`: `packed_hidden_states`, `packed_positions`, `unpacking_mask`, `active_mask`.
- `MoSRAHLayer` passes `unpacking_mask` (not `active_mask`) to `unpack_experts`.
- `MoSRAHLayer` passes `active_mask` (packed semantic liveness) to BEA.
- All-live behavior remains semantically equivalent to pre-blocker behavior.

**What should stay unchanged:**
- `MoSRAHRouter`, `pack_experts`, `unpack_experts`, and BEA internals are not modified.
- `MoSRAHLayer` does not gain new attention semantics; it remains an orchestration layer for its sub-components.
- This unit does not thread the active mask above `MoSRAHLayer`; that is 14.I.

**Tests delta:**
- Add coverage that dead outer tokens do not affect `load_balance_loss` or `max_vio` at the `MoSRAHLayer` boundary.
- Add coverage that dead outer tokens produce suppressed outputs from `MoSRAHLayer` (BEA receives the correct packed active mask).
- Add all-live equivalence coverage against pre-blocker `MoSRAHLayer` behavior.

**Preliminary notes:**
- Keep this unit strictly to the three broken call sites in `mosrah.py`.
- The four-value unpack from `pack_experts` — `(packed_hidden_states, packed_positions, unpacking_mask, active_mask)` — is the key structural change; verify the names are not swapped.

### Unit 14.I (Blocker) — Mask Plumbing Through SHRAMHybridLayer, DecoderLayer, and ShramModel

**Change delta from 14.B–14.H:** Thread the current-chunk active mask through the remaining SHRAM orchestration layers so both attention paths receive it at runtime.

**Context of Correctness**

After 14.C–14.H, all sub-units have mask-aware contracts. The remaining work is pure interface plumbing: the same current-chunk mask must be accepted at the `ShramModel` boundary and forwarded unchanged through `DecoderLayer` and `SHRAMHybridLayer` to both the local path and the (now mask-wired) `MoSRAHLayer`. No new semantics are introduced at these orchestration layers.

**New / changed contract:**
- `ShramModel.forward` accepts `active_mask: torch.Tensor` of shape `(B, N)` as a required parameter.
- `ShramModel` forwards `active_mask` to each `DecoderLayer`.
- `DecoderLayer.forward` accepts and forwards `active_mask` to `SHRAMHybridLayer`.
- `SHRAMHybridLayer.forward` accepts `active_mask` and passes it to both the local attention path and `MoSRAHLayer`.
- The same current-chunk active mask reaches both the local path and the sparse path.
- No new attention semantics are introduced at these orchestration layers.
- Existing all-live behavior remains semantically equivalent to pre-blocker behavior.

**What should stay unchanged:**
- SHRAM hybrid semantics remain `H(x) = h_l(x) + h_s(x)`.
- Decoder residual / pre-norm structure remains unchanged.
- `MoSRAHLayer`, `SlidingWindowAttention`, `MoSRAHRouter`, `pack_experts`, and BEA internals are not modified.

**Tests delta:**
- Add coverage that the same active mask reaches both the local path and the sparse path through `SHRAMHybridLayer`.
- Add all-live equivalence coverage for the orchestration path end to end.

**Audit requirements**
- Inspect `SHRAMHybridLayer.forward` and verify that the same `active_mask` is passed to both `self.local_attention(...)` and `self.sparse_attention(...)` — not two independently constructed masks.
- Inspect `DecoderLayer.forward` and verify that `active_mask` is forwarded to `self.attention(...)` without modification.
- Inspect `ShramModel.forward` and verify that `active_mask` is passed to every `layer(...)` call in the decoder loop without modification.

**Preliminary notes:**
- Keep this unit narrow: call-signature updates and forwarding only.
- Prefer explicit required parameters over optional defaults — the active mask is always required.
- Note that `DecoderLayer` already has a `mask` parameter that is received but not forwarded; determine whether it is this active mask or a different artifact, and resolve cleanly.
- If implementation reveals any further boundary requiring the mask, surface it rather than silently expanding scope.

### Unit 15.A

SHRAM supports both tied and untied embedding configurations by config. Because HuggingFace save/load behavior needs to know whether a given model instance actually uses tied weights, this decision cannot be treated as a fixed class-wide fact.

**Procedures**

- Probed the HuggingFace save/load path directly.
- Checked whether pretrained loading preserves preexisting tensor aliasing.
- Checked where HuggingFace records and reads tied-weight declarations.
- Checked whether constructor-time config rebuilding occurs before checkpoint loading.

**Findings**

- HuggingFace during pretrained-model load may overwrite parameter objects rather than merely filling existing tensor storage, and save/load bookkeeping must know about intentional ties.
- Although pretrained loading does construct the model from config first, later load steps may replace parameters, so constructor-created aliasing is not by itself a complete serialization guarantee.
- Therefore the only safe action is:
  - if tied embeddings are enabled, tie the embeddings and store the tied declaration at `_tied_weights_keys`
  - if tied embeddings are disabled, do neither

### Unit 15.B — SHRAM CausalLM Output

`ShramForCausalLM` must expose a HuggingFace-causal-LM-compatible wrapper output while also truthfully surfacing SHRAM-specific monitoring and auxiliary values produced by the backbone.

**Invariants this unit must satisfy:**

- The wrapper output preserves the standard causal-LM boundary information expected by HuggingFace-facing consumers:
  - logits,
  - past_key_values,
  - hidden_states,
  - loss when labels are provided.
- The wrapper output also exposes:
  - `load_balance_loss` as a scalar auxiliary training value,
  - `max_vio` as a detached scalar monitoring value.
- `load_balance_loss` and `max_vio` are surfaced truthfully from the backbone result and are not recomputed or semantically altered in the wrapper.
- The output type is SHRAM-specific and exists to extend the standard causal-LM wrapper output with these additional fields.
- The output remains compatible with normal HuggingFace-facing access patterns for causal-LM outputs.

**Tests:**

- Verify the output exposes the standard causal-LM wrapper fields.
- Verify the output exposes `load_balance_loss` and `max_vio`.
- Verify `load_balance_loss` is finite when present.
- Verify `max_vio` is finite when present and detached.

**Audit:**

- Verify the output type exists only to extend the wrapper boundary truthfully and does not move backbone semantics upward into the wrapper.
- Verify `load_balance_loss` and `max_vio` are passed through from `ShramModel` rather than recomputed.

**Preliminary implementation strategy:**

- Introduce a small SHRAM-specific output type that extends the standard causal-LM output shape with `load_balance_loss` and `max_vio`.
- Keep this unit narrow: output contract only, no wrapper-behavior changes here.
- 
### Unit 15.C (Blocker) — Top-Level Sequence Length for HuggingFace Generation

**Responsibility:** Restore truthful top-level sequence-length reporting for the SHRAM cache stack so built-in HuggingFace generation can operate against `ShramCache`.

**Context of Correctness**

The HuggingFace decoding system, even in greedy modes, requires the ability to tell how many tokens have previously been generated. It asks this through `get_seq_length()`, which is currently configured to throw an error.

We previously chose to throw because ragged-batch cache state seemed to prevent a truthful scalar report. That was a misunderstanding of what HuggingFace is asking for here. It does not need the number of currently active tokens in the cache; it needs the total sequence length processed so far for the current sequence. The local sliding-window path now provides a simpler place to track that quantity truthfully.

Because built-in generation depends on this value, Unit 15.C cannot be completed correctly until the cache stack reports it.

**Invariants this unit must satisfy:**

- `LocalSlidingWindowLayerCache` tracks the cumulative total number of tokens processed for the current sequence.
- That tracked quantity is total processed sequence length, not active-token count and not current window occupancy.
- `ShramLayerCache.get_seq_length()` reports this value truthfully for the layer.
- `ShramCache.get_seq_length()` reports this value truthfully at the top-level HuggingFace cache boundary.
- This sequence-length report is compatible with ordinary HuggingFace generation usage.
- Existing cache semantics for masking, ragged expert-state handling, and local/sliding-window behavior are otherwise unchanged.
- The design does not infer sequence length from MoSRAH ragged occupancy.

**Tests**

- Add or update tests for `LocalSlidingWindowLayerCache` to verify cumulative processed-token count across repeated updates.
- Add or update tests for `ShramLayerCache` to verify `get_seq_length()` reports the local cache’s cumulative processed-token count.
- Add or update tests for `ShramCache` to verify top-level `get_seq_length()` reports the truthful model-wide sequence length.
- Add or update HuggingFace wrapper/generation tests as needed so the previous `get_seq_length()` failure path is no longer hit during ordinary generation.

**Preliminary implementation strategy:**

- Prefer tracking cumulative processed tokens directly in `LocalSlidingWindowLayerCache` rather than reconstructing sequence length indirectly from other cache state.
- Source `ShramLayerCache.get_seq_length()` from the local/sliding-window side, where total sequence progress is naturally defined.
- Source `ShramCache.get_seq_length()` by forwarding the truthful per-layer value exposed at the layer-cache boundary.
- If no contradictory ground truth appears in the code, avoid involving MoSRAH occupancy or active-mask semantics in this scalar sequence-length concept.

### Unit 15.D — ShramForCausalLM

**Responsibility:** Define and verify the top-level HuggingFace-facing causal language model boundary for SHRAM. This unit owns the token embedding, LM head, wrapper-level causal-LM loss behavior, HuggingFace generation/cache orchestration at the wrapper boundary, and translation between HuggingFace-facing token inputs/outputs and the delegated `ShramModel` backbone.

**Context of Correctness**

`ShramForCausalLM` is the huggingface interface boundary. It is responsible for best satisfying and ensuring the satisfaction of huggingface expectations and standards. Huggingface expects token embedding lookup, vocabulary projection, next-token loss handling, weight tying, wrapper-level cache construction/resolution, wrapper-level positional-generation wiring, and truthful exposure of backbone-produced auxiliary or monitoring values to downstream HuggingFace-facing consumers to be on this unit. It also expects to be able to pass in positional ids if needed during generation, a whole-sequence mask, and preferably would not use the cache location feature as that is becoming obsolete. 


**Invariants this unit must satisfy:**

- `ShramForCausalLM` is the HuggingFace-facing causal-LM wrapper for SHRAM and owns the token embedding and LM head.
- The unit accepts token IDs at the wrapper boundary, converts them to embeddings for `ShramModel`, receives backbone hidden states, and projects them to vocabulary logits at the causal-LM boundary.
- The unit computes wrapper-level causal next-token cross-entropy loss when labels are provided. The causal shift is applied here rather than required from the caller.
- The unit preserves the delegated role of `ShramModel`: transformer computation and backbone-produced internal semantics are not recomputed or reinterpreted here.
- The forward output satisfies the HuggingFace causal-LM wrapper boundary while also exposing SHRAM-specific `load_balance_loss` and `max_vio`.
- `load_balance_loss` is present in the forward output as a scalar auxiliary training value. The training loop is responsible for the paper's scaling weight; this model does not apply it.
- `max_vio` is present in the forward output as a detached scalar monitoring value. It is the layer-maximum value produced by `ShramModel` and is not modified here.
- The wrapper preserves standard wrapper responsibilities such as weight tying, output-head exposure, and compatibility with the expected HuggingFace causal-LM wrapper interface, unless a behavior is explicitly excluded.
- The unit owns cache resolution at the HuggingFace wrapper boundary. It also owns mask slicing in terms of the relevant recent sequence, as described in 14.B.
- When cached execution or generation requires internal cache construction at this boundary, the constructed cache is a `ShramCache`, not a plain `DynamicCache`.
- A caller who supplies an explicit `ShramCache` as `past_key_values` has that cache used unchanged.
- `use_cache=True` is supported at this wrapper boundary. Not using cache does not genrate a ShramCache
- Unsupported HuggingFace-facing behavior must fail explicitly rather than being silently accepted with wrong semantics.
- Tied embeddings are supported. They are both directly tied for immediate usage, and declared as instance variables on the huggingface instance field for saving and loading purposes.
- Tests and design for this unit are structured so the unit can be trusted as a verified black box by downstream training, generation, and hub-facing consumers.

**Tests:**

Training unit:
- Verify the forward output exposes `load_balance_loss` and that it is a finite scalar.
- Verify the forward output exposes `max_vio`, that it is a finite scalar, and that it is detached.
- Verify the wrapper still produces logits of shape `(B, N, vocab_size)`.
- Verify wrapper-level causal-LM loss is present when labels are provided and absent when labels are not provided.
- Verify wrapper-level causal-LM loss still applies the correct next-token shift.
- Verify tied-embedding behavior remains correct when configured.
- Verify torch dynamo compiles, or discuss verifying it on the backbone level

Inference unit:

- Verify `generate()` may be called successfully in ordinary token generation.
- Verify `generate()` may be called successfully in beam search.
- Verify `generate()` may be called successfully in direct reconstructive / contrastive search (or discuss removing it if unsupported).
- Verify `generate()` called with an explicit `ShramCache` uses that cache.
- Verify `generate()` may be called with a sequence of raggedly batched input prompts, and this successfully operates. 
- Verify unsupported HuggingFace-facing behaviors fail explicitly rather than being silently accepted.
- Verify 

**Audit**

- Verify the wrapper performs token/interface translation and causal-LM boundary duties here, while delegating transformer computation and backbone semantics to `ShramModel`.
- Verify embeddings are created at this boundary and logits are produced at this boundary rather than inside `ShramModel`.
- Verify wrapper-level label shifting and loss computation occur here rather than being delegated downward or expected from the caller.
- Verify internally constructed caches at the cached-forward / generation boundary are `ShramCache` objects rather than generic HuggingFace dynamic caches.
- Verify `load_balance_loss` and `max_vio` are exposed truthfully from the backbone result rather than recomputed or semantically altered here.
- Verify commonly used HuggingFace-facing support has been evaluated and is either implemented truthfully or explicitly excluded.
- Verify any existing wrapper behaviors that remain unsupported, such as behaviors with no truthful SHRAM boundary meaning, raise explicitly and do not silently degrade.

**Preliminary implementation strategy:**

- Treat this as a wrapper-boundary update, not a backbone redesign.
- Preserve the existing HuggingFace wrapper structure wherever it remains truthful, and extend it only where SHRAM-specific cache and output requirements demand it.
- Make sure to audit how huggingface ideomatically wants to be implemented and use the correct version. If this means a major refactor, discuss it and insert it as a blocker (one unit, 14, has already been done in this way.)
- `ShramModel` should continue to receive pre-embedded hidden states rather than token IDs.
- The wrapper should continue to own embedding lookup, LM-head projection, label-shifted CE loss, and weight tying.
- The wrapper should continue to support any needed functionality such as reinitializing or reshaping heads.
- The likely override point for ensuring `generate()` constructs a `ShramCache` is `_prepare_cache_for_generation(generation_config, model_kwargs, generation_mode, batch_size, max_cache_length)`. If the ground situation shows a different HuggingFace hook is now responsible, that should be surfaced rather than worked around silently.
- Other such generation or preparitory helper functions should be examined for any needed changes. 
- Prefer runtime tests for real wrapper behavior and use audit for structural delegation and cache-construction facts that are better checked by direct inspection than by brittle implementation-shaped tests.
- If HuggingFace-facing support obligations are discovered that are not yet truthfully satisfied, surface them as blockers rather than filling gaps silently.

### Unit 16 — upload_to_hub.py

**Responsibility:** Publish the SHRAM architecture and tokenizer to the HuggingFace Hub so a researcher can instantiate a freshly initialized model with no local setup beyond `pip install transformers`.

**Context of Correctness:** The upload script was written for the Llama 3 baseline and has not been adapted for SHRAM. It references wrong parameter names, presents default values under framing that implies a pretrained checkpoint rather than a configurable architecture, and targets the wrong repository and path. A researcher following the published usage instructions against the current Hub state would get errors or a misleading model card.

**Invariants this unit must satisfy:**

- Architecture files are uploaded under the `architecture_core` path within the repository.
- No weight files are uploaded.
- The model card accurately identifies this as the SHRAM architecture, not Llama.
- The model card presents constructor default values explicitly framed as overridable defaults, not as the parameters of a pretrained model.

**Preliminary implementation strategy:** Three files need changes. In `upload_to_hub.py`: update `REPO_ID` to `smithblack-0/SHRAM`, add `path_in_repo="architecture_core"` to the `upload_folder` call, and replace the Llama field names in `_render_config_table` with the SHRAM field names from `ShramConfig`. In `model_card.md`: rewrite for SHRAM — correct the title, remove the `llama` tag, update the license section, and reframe the defaults table section header and surrounding copy to make clear these are overridable constructor defaults, not pretrained model parameters.

---

### Unit 17 — Documentation

**What:** Write `documentation.md` covering design decisions, deviations from the paper, and
limitations. Update `README.md` with accurate architectural details. Record every open decision
resolved during implementation and the rationale for each.

**Invariants this unit must satisfy:**
- Every open decision resolved during implementation is recorded with its rationale.
- Limitations are documented explicitly, including any known train/inference mismatches.
- The model card accurately describes the SHRAM variant.

---

### Unit 18 — End-to-End Tests

**What:** Full-stack smoke tests: instantiate from config, run a training step, verify loss
decreases. Include load-balance loss in the training step. Include network tests for the Hub
round-trip.

**Invariants this unit must satisfy:**
- The model can be instantiated from a config, run a forward pass, compute loss, and backpropagate
  without error.
- The load-balance loss is accessible from the forward output and participates in the backward pass.
- Network tests verify the Hub round-trip.

---

### Unit 19 — Final Audit

**What:** Review every file in `src/shram/` against the invariants in `job.md`. Verify no
hardcoded values, no missing documentation, no gaps between tests and intent. Apply the
close-the-testing-gap rule to any defect found.

**Invariants this unit must satisfy:**
- Every invariant in job.md is satisfied and has a corresponding test.
- No file has hardcoded architectural parameters.
- All documentation standards are met.
- Every decision recorded in this plan has a corresponding entry in `documentation.md`.
