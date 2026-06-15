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
- [X] Unit 17 — documentation
- [X] Unit 18.A (Blocker) — Combined loss refactor: ce_weight + load_balance_weight forward args
- [X] Unit 18.B (Blocker) — End-to-End Subfolder Blocker: branch-based upload replacing subfolder approach
- [X] Unit 18.C (Blocker) — End-to-End Subfolder Blocker: basic approach. Work this time dammit.
- [X] Unit 18.D (Blocker) — Huggingface, bafflingly, does not support folders. Fix it without loosing the important organization of the project.
- [X] Unit 18.E — end-to-end tests
- [X] Unit 19.A — Devops Concerns
  - [X] Unit 19.A.1 — Standardize upload infrastructure across all model folders
  - [X] Unit 19.A.2 — Local dev environment files per model
  - [X] Unit 19.A.3 — GitHub Actions workflows
- [X] Unit 19.B — ShramConfig: explicit inference_sequence_length parameter
- [X] Unit 19.C (Blocker) — Expose total MoSRAH layer parameter count
- [X] Unit 19.E (Blocker) — E2E torch-dynamo compile coverage gap
- [X] Unit 19.F (Blocker) — SlidingWindowAttention torch.compile failure
- [X] Unit 19.F.1 — Position-zero constraint: compile-compatible enforcement (revised)
- [X] Unit 19.F.2 — Compile-time error for missing capture_scalar_outputs
- [X] Unit 19.G (Blocker) — RotaryEmbedding: preallocate to maximum_sequence_length, eliminate .item()
- [X] Unit 19.G.0 (Blocker) — Expert packing: interface consolidation
- [X] Unit 19.G.1 (Blocker) — Expert packing: static T preallocation and overflow detection
- [X] Unit 19.G.2 (Blocker) — Expert packing: replace _bincount_rows with scatter_add
- [X] Unit 19.G.3 (Blocker) — Load balance frequency aggregation: p-mean and load_balance_p
- [X] Unit 19.G.4 (Blocker) — Restore mask symmetry for compiled inference via create_masks_for_generate override
- [X] Unit 19.G.5 (Blocker) — Static cache rebuild for compiled inference
- [X] Unit 20.A (Blocker) — Release pipeline: dev repository staging and E2E test gate
- [X] Unit 20.B — stage_for_hub.py: inject explicit import block into staged huggingface.py
- [X] Unit 21 — Capacity issues
- [X] Unit 21.A — huggingface initialization fix
- [X] Unit 21.B — training capacity fix. (Sinkhorn implementation — SUPERSEDED by Unit 21.C due to convergence failure)
- [X] Unit 21.C — Bidding-based capacity enforcement
- [X] Unit 22.A — fix hacked assertions using torch._assert_async.
- [X] Unit 22.B — Remove capture_scalar_outputs dependency.
- [X] Unit 22.C — Expert packing: fixed-shape compact-to-padded transfer.
- [X] Unit 23.A — Compiled inference-style execution: eval, no_grad, mixed-precision coverage
- [X] Unit 23.B — Fix position ID resolution: replace cumsum with arange + active-token bias
- [X] Unit 23.C — Documentation: compile-mode constraints and minor documentation gaps
- [X] Unit 23.D — Router diagnostics: refactor return signature + add load-balance health scalars
- [X] Unit 24.A — Load balance loss: replace DeepSeek fixed-step mechanism with log-probability auxiliary loss
- [X] Unit 24.B — Routing logit variance reduction: near-zero scalar gate
- [X] Unit 24.C — Biased routing probabilities: incorporate expert_bias into P
- [X] Unit 25 — Load balancing rebuild
- [X] Unit 25.A — Load balancing loss fix
- [X] Unit 25.B — Balancing offset mechanism rebuild
- [X] Unit 25.C — Integral Routing
- [X] Unit 26.A — Restore Coupled Routing
- [X] Unit 26.B — Temporal Overcapacity Loss
- [X] Unit 26.C — Temporal Overcapacity Integration
- [X] Unit 27 — Causal Overcapacity Loss
- [X] Unit 28 — Mechanical Load Balancing
- [X] Unit 29.A — Split residual gates in DecoderLayer (vector)
- [ ] Unit 29.B — Scalar residual gates
- [ ] Unit 29.C — Config-selectable residual gate vs fixed scale
- [ ] Unit 30 — Final Audit

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

**Responsibility:** Define and verify the documentation surface needed for a user or maintainer to understand, control, use, and continue maintaining the SHRAM portion of the repository in support of the corresponding SHRAM/SRAM-effect paper.

**Context of Correctness**

The SRAM effect is a hypothesized trade that lets certain classes of models trade parameters for long-sequence performance at a linear asymptotic rate; details are available in the corresponding paper. Probing this effect required the construction of a complex and unusual SHRAM subsystem inside a broader multi-model repository. Keeping that SHRAM portion of the repository usable and understandable, while preserving its connection to the paper and supporting ongoing maintenance, is therefore a critical part of correctly studying the effect.

**Invariants this unit must satisfy:**

- The documentation explains, at a brief operational level, what the SHRAM portion of the repository is for and how it relates to the corresponding paper.
- The documentation makes the connection to the paper explicit and easy to find.
- The documentation tells a user how to instantiate and use the SHRAM portion of the repository at its intended surface, rather than forcing them to infer basic usage from source code.
- The documentation identifies the main SHRAM control surfaces relevant to operating or probing the system, without trying to restate the full theory from the paper.
- The documentation identifies the major caveats or unusual usage constraints a 1user must know in order to use the SHRAM portion of the repository correctly.
- - The documentation provides sufficient context for ongoing maintenance of the SHRAM portion of the repository.
- The documentation explains the relevant software-artifact surface: where the important SHRAM entrypoints are, how this part of the repository is structured at a high level, and how new versions are uploaded or exposed.
- The documentation defers deeper theory, proofs, and detailed architectural justification to the paper rather than attempting to duplicate them.
- The documentation is sufficient for a technically competent reader to understand how to operate and maintain the SHRAM portion of the repository in support of the paper.

** Audit**

- Verify a reader can identify what the SHRAM portion of the repository is for and how it relates to the paper from the documentation alone.
- Verify the paper is linked or otherwise made easy to find from the SHRAM documentation surface.
- Verify a reader can find the main SHRAM usage entrypoints without reading implementation files first.
- Verify a reader can find the major SHRAM control knobs or operational surfaces relevant to running or probing the system.
- Verify major SHRAM caveats and unusual usage constraints are documented.
- Verify the upload/versioning path for SHRAM is documented or linked from the relevant documentation surface.
- Verify the documentation makes clear how SHRAM sits alongside the repository’s other model implementations.
- Verify the documentation provides enough context that future maintenance of the SHRAM section does not require reconstructing project intent only from code or old plan history.
- Verify the documentation does not try to reproduce the paper’s deeper theory where a short explanation plus a pointer is more correct.

**Preliminary implementation strategy**

- Use the top-level repository documentation as the main user-facing operating surface, with SHRAM-specific documentation attached to the SHRAM portion of the repo.
- Explain briefly what the SRAM effect / SHRAM system is trying to probe, then point outward to the paper for theory and deeper architectural justification.
- Document the practical SHRAM surface: structure, usage entrypoints, control knobs, caveats, maintenance-relevant context, and upload/version workflow.
- Document SHRAM as one model family within a broader repository rather than as a standalone project.
- Prefer short operational explanations plus links over reproducing long theoretical sections from the paper.
- If possible, make the paper directly accessible from the SHRAM documentation surface so the repository and paper remain visibly connected.
---

### Unit 18.A (Blocker) — Combined loss refactor: ce_weight + load_balance_weight forward args

**What:** Refactor `ShramForCausalLM.forward()` so that `loss` follows the HuggingFace MoE
convention: the combined weighted total of CE loss and load balance loss. Expose the individual
components as `ce_loss` and `load_balance_loss` on the output for logging.

**Why this is a blocker:** HuggingFace's `Trainer` and any standard training loop calls
`.backward()` on `out.loss` only. With the current design, the router's `expert_bias` and
load-balance-relevant parameters never receive gradients in standard training. This violates
the convention established by Mixtral and other HuggingFace MoE models.

**Invariants this unit must satisfy:**
- `forward()` accepts `ce_weight: float = 1.0` and `load_balance_weight: float = 0.01` as
  explicit keyword arguments. Defaults match the paper's recommendations.
- When labels are provided, `out.loss = ce_weight * ce_loss + load_balance_weight * load_balance_loss`.
- When labels are not provided, `out.loss` is `None`.
- `out.ce_loss` contains the raw unweighted cross-entropy loss (or `None` if no labels).
- `out.load_balance_loss` continues to contain the raw unweighted load balance loss.
- `out.max_vio` is unchanged.
- `expert_bias` and all other load-balance-graph parameters receive gradients when
  `out.loss.backward()` is called with labels provided.
- `ShramCausalLMOutput` gains a `ce_loss` field. All existing fields remain.
- HuggingFace's `Trainer` works correctly with default weights (no custom kwargs required).

**Research note:** Weights are forward arguments, not config fields. Researchers who want
non-default weighting must use a custom training loop or `Trainer` subclass — this is
acceptable and consistent with how auxiliary loss weighting is handled in the literature.

---

### Unit 18.B - End-to-End Subfolder Blocker

**Responsibity**: Perform a minor amount of refactoring to allow the storage of multiple revisions rather than multiple files

**Context of Correctness**

Huggingface initially appeared to support loading from subfolders located at a remote repository. This was not in fact true. While it is the case you can load any folder locally simply by naming the directory, the "subfolder" field of the huggingface load kwargs is not correctly accounted for by the load module subcode.

Instead, the most viable way to keep multiple models in the same repository is to form multiple branches and use the revision tag to load them. It is possible for instance to just upload the folder to the root directly, in different branches, to maintain separation

**Invariants**

- Code and test among files **upload_to_hub.py*, **test_upload_to_hub.py** is cleaned up to no longer expect the existance of an architecture_core folder.
- The upload system now selects a branch feature instead of a subfolder feathre
- The upload system now makes a new branch when needed before uploading the folder as well
- The folder is uploaded directly into the repository root. 
- For consistency, the current revision will be called "primary". Loading is executed by passing in a revision indicator.
- Downstream tests are modified as needed to express these invariants. 


**Preliminary implementation strategy**

- Before upload, check whether the target branch exists; create it if it does not.
- Upload the architecture folder directly to the repository root of that branch.
- Load by passing both `revision` and `code_revision`.

Example usage:

```python
config = AutoConfig.from_pretrained(
    "smithblack-0/SHRAM",
    revision="primary",
    code_revision="primary",
    trust_remote_code=True,
)

model = AutoModelForCausalLM.from_config(
    config,
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(
    "smithblack-0/SHRAM",
    revision="primary",
)
```

### Unit 18.C - Huggingface Subfolder Blocker (v2)

**Responsibity**: Perform a minor amount of refactoring to ensure we only store the architecture itself in the main system.

**Context of Correctness**

It turns out while AutoModelForCausalLM.from_config permits the injection of a code revision, it in no way responds to it. This makes the system currently in use pointless. The only option is to fall back to a raw repository. We update to main directly.
**Invariants**

- Code and test among files **upload_to_hub.py*, **test_upload_to_hub.py** is cleaned up to no longer expect the existance of a revision.
- The upload system now just uploads directly in a standard manner
- The folder is uploaded directly into the repository root. 
- Downstream tests are modified as needed to express these invariants. 
- Documentation.md is modified to reflect the correct strategy. 
- e2e tests is modified to reflect the strategy. These do not have to be ru

**Preliminary implementation strategy**

- This should be straightforward 

Example usage:

```python
config = AutoConfig.from_pretrained(
    "smithblack-0/SHRAM",
    trust_remote_code=True,
)

model = AutoModelForCausalLM.from_config(
    config,
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(
    "smithblack-0/SHRAM",
)
```

### Unit 18.D — Huggingface Upload System Rewrite

**Responsibility**: Rewrite the upload system to make a temporary staging folder of flattened files to handle limitations of the huggingface system

**Context of Correctness**

Custom Huggingface automodel uploads are unable to contain any folders. Deep in the internal's of huggingfaces 'dynamic_module_utils.py' package the line "    modules_needed = check_imports(resolved_module_file)" is used immediately to represent the modules needed, when those modules may be a multiple-leveled relative import such as "cache.shram_cache.py". This is never resolved to "cache/shram_cache.py" before file fetching or uploading begins, making huggingface foundationally unable to handle folders.  This is a significant issue. 

There are several lines of possible fixes:

* Flattening the repository is possible. However, We loose all organization and become unmaintainable to an outsider 
* Switching to a package approach is possible. However, this late into development it is not a good idea, and the other coresident models are not build for this approach anyhow

The only solution of any level of correctness is thus a well-written staging approach that makes a staging folder that is flattened, and refactors imports as needed to make this work.

**Invariants**

* The upload script is modified to execute a flattening 'staging' approach wherein paths such as "cache/shram_cache.py" become "__cache__shram_cache.py". This produces a sorted flat folder for huggingface to work from. 
* libcst is used during this process to refactor source code to use single-level relative imports rather than multi-level folder imports, to avoid breaking code imports. 
* The system is well-written and modular following best practices using helper functions

**Tests**

* The ability to import from the staged files must be tested at some point; it is not viable to upload something that may not successfully import

**Preliminary implementation strategy**

* It is likely going to be easiest to meet all criteria to have a separate refactoring system that takes a location to make a folder at, and then invoke that in the main script. This allows testing of imports independently of the primary logic.


### Unit 18.E — End-to-End Tests

**What:** Full-stack smoke tests: instantiate from config, run a training step, verify loss
decreases. Include load-balance loss in the training step. Include network tests for the Hub
round-trip.

**Invariants this unit must satisfy:**
- The model can be instantiated from a config, run a forward pass, compute loss, and backpropagate without error.
- The load-balance loss is accessible from the forward output and participates in the backward pass.
- Network tests verify the Hub round-trip.

---

### Unit 19.A — DevOps concerns.

**Responsibility**: Setup the proper devops concerns for automated updating and testing

**Context of Correctness**:

While it is in theory possible to continue to use the repository to manually update systems, it is likely far more correct to create a proper github-supported workflow that may run tests as needed, has protected branches, and will automatically upload passing models to the hub.

**Invariants**

* The github system now has branch protection; we work using a checked out branch instead
* The github system is configured to automatically search for model folders and run the associated actions and pull request merges
* On pull request all tests have to pass for merge to be allowed.
* Pull requests may have a 'Release' tag attached to them in github. When done, tests are run then each relevant upload_to_hub script runs too.
* The upload_to_hub system is redesigned to use github secrets. These secrets are unique to each script.
* A setup_dev_environment.py file that runs a requirements file should be located in the folder for that model. When run, it would set up any needed requirements.
* A record_requirements.py file that will write a requirements file into the main shram folder for usage by setup_dev_enviroment.

---

### Unit 19.A.1 — Standardize upload infrastructure across all model folders

**Responsibility:** Bring all three model upload systems (llama3, mosa, shram) to a common standard so the orchestrator can treat every model identically and the repository is forkable.

**Context of Correctness:** This is a single-system repository. Professional quality means a contributor who forks it and adds a new model finds a clear, consistent pattern. SHRAM has a staging system and secrets-ready design; llama3 and mosa use an older pattern. That inconsistency is a quality and maintainability defect. Standardizing here is what makes 19.A.3 (the orchestrator) possible — the orchestrator can only call each model's upload script uniformly if the scripts present a uniform interface.

**Invariants:**
- Every model folder contains a `stage_for_hub.py` with a `stage(source_dir, dest_dir)` function having the same signature as SHRAM's
- Every `upload_to_hub.py` reads its HuggingFace token from an environment variable unique to that model (e.g. `SHRAM_HF_TOKEN`, `LLAMA3_HF_TOKEN`, `MOSA_HF_TOKEN`) rather than prompting interactively
- Every `upload_to_hub.py` with `REPO_ID = None` exits cleanly with an informative message without attempting any upload or raising an exception
- Every `upload_to_hub.py` stages through `stage_for_hub.py` into a temporary directory before uploading, never uploading the source tree directly

**Tests:**
- Each `upload_to_hub.py` invoked with `REPO_ID = None` exits without error
- Each `stage_for_hub.py` is importable and exposes a `stage(source_dir, dest_dir)` callable

**Preliminary implementation:** Port SHRAM's `stage_for_hub.py` and updated `upload_to_hub.py` pattern to llama3 and mosa. Where a model's source tree is already flat, `stage_for_hub.py` may be a thin wrapper that copies files without renaming — the interface is what matters, not the complexity of the implementation.

---

### Unit 19.A.2 — Per-model pyproject.toml and dev environment setup

**Responsibility:** Each model folder declares its own dependencies and provides a one-command local install so contributors and CI can reproduce a working environment for that model independently.

**Context of Correctness:** Different models may have different dependencies — conflating them into a single repo-wide file is a maintainability defect and prevents per-model isolation. The correct unit of dependency declaration is the model folder. A `pyproject.toml` per model is the standard Python mechanism for this, requires no external tooling, and is hand-maintainable — CI will surface missing entries when imports fail, so there is no need for a generation script.

**Invariants:**
- `src/llama3/pyproject.toml`, `src/mosa/pyproject.toml`, and `src/shram/pyproject.toml` exist and each declares that model's runtime dependencies under `[project.dependencies]`
- `src/llama3/setup_dev_environment.py`, `src/mosa/setup_dev_environment.py`, and `src/shram/setup_dev_environment.py` exist, are runnable from the repository root without arguments, and install that model's dependencies via `pip install -e`
- After running `setup_dev_environment.py` for shram, `from shram.model.huggingface import ShramForCausalLM` is importable; same pattern holds for llama3 and mosa under their respective package names
- Each model's `documentation.md` documents both the HuggingFace path and the local install option

**Tests:**
- `setup_dev_environment.py` is present and importable in each model folder
- `setup_dev_environment.py` exposes a `setup()` callable
- After install, each model's top-level model class is importable under its short package name

**Preliminary implementation:** `pyproject.toml` uses `[project]` for metadata and dependencies, `[tool.setuptools.packages.find]` with `where` pointing at the src layout to locate the package. `setup_dev_environment.py` resolves the model folder path relative to `__file__` and calls `subprocess.run([sys.executable, "-m", "pip", "install", "-e", str(model_dir)], check=True)`. The installed short package names are `shram`, `llama3`, and `mosa` respectively.

---

### Unit 19.A.3 — GitHub Actions workflows

**Responsibility:** Automate testing on every PR and Hub upload on Release-labelled merges via a modular workflow structure.

**Context of Correctness:** Manual upload and manual test runs do not scale and are not enforced. Branch protection is only meaningful if there is a CI system backing it. The workflow structure must be modular — one orchestrator, one reusable workflow file per model — so adding a new model means adding one file and one orchestrator registration, nothing else.

**Invariants:**
- `.github/workflows/orchestrator.yml` exists, triggers on PR and on Release-labelled merge to master, and calls each model's workflow file
- `.github/workflows/llama3.yml`, `shram.yml`, and `mosa.yml` exist as reusable workflows; each owns its own test path, secret name, and upload invocation
- On every PR: all model tests must pass or the PR cannot be merged
- On Release-labelled merge: tests run first; upload runs per model only if tests pass; a model with `REPO_ID = None` is skipped without failing the workflow
- Each model workflow installs dependencies via `pip install -e src/<model>/` before running tests
- Each model workflow reads its HuggingFace token from a GitHub secret scoped to that model

**Tests:** Not locally unit-testable. Verified by opening a test PR and observing workflow execution on GitHub.

**Preliminary implementation:** Orchestrator uses GitHub Actions `workflow_call` to invoke each model's reusable workflow. Each model workflow has two jobs: `test` (runs `pip install -e src/<model>/` then pytest for that model's test folder) and `upload` (conditional on Release label, `needs: test`). Branch protection rules are configured manually in the GitHub repository settings — this is not automatable from the codebase itself and must be documented as a setup step.

### Unit 19.B — ShramConfig: explicit `inference_sequence_length` parameter

**Responsibility:** Replace the implicit/helper-based `inference_sequence_length` mechanism with an explicit optional constructor parameter.

**Context of Correctness:** LLM test harness runners initialize models via config-override at construction time. A parameter that can only be set post-construction via a helper method is invisible to that paradigm and cannot be overridden. The kwargs catch-all that currently makes this "work" accidentally is a silent footgun — it accepts arbitrary typo'd kwargs without error.

**Invariants:**
- `inference_sequence_length` is an explicit optional constructor parameter with default `None`; when `None`, it is set to `training_sequence_length` at construction time
- `set_inference_context()` does not exist on `ShramConfig`
- No kwargs catch-all for `inference_sequence_length` exists; it is declared explicitly in the signature
- Non-positive values passed at construction raise `ValueError`
- `scale` property continues to return `inference_sequence_length / training_sequence_length`
- An explicitly passed `inference_sequence_length` survives a `to_dict` / `from_dict` roundtrip
- Tests verify: default equals training length, explicit value stored, non-positive raises `ValueError`, roundtrip preservation, `set_inference_context` does not exist as an attribute

**Preliminary implementation:** Remove `set_inference_context()` from `configuration.py`. Change the kwargs pop to an explicit named parameter defaulting to `None`. Validation moves to construction time. `test_configuration.py` tests for `set_inference_context` are replaced with constructor-based equivalents.

---

### Unit 19.C (Blocker) — Expose total MoSRAH layer parameter count

**Responsibility:** Provide a public method on `ShramForCausalLM` that returns the total number of
trainable parameters belonging to MoSRAH layers (routing, BEA projections, expert bias) across all
decoder layers, for use in experimental plotting.

**Why this unit, why here:** The experiment consuming this model needs the MoSRAH parameter count
to construct plots comparing parameter count against performance. This is a read-only introspection
capability with no architectural side effects. It is a blocker before the final audit because the
audit certifies the public interface is complete.

**Invariants this unit must satisfy:**
- The returned count equals the sum of `p.numel()` for all parameters belonging to MoSRAH
  submodules across all decoder layers — not the full model, not the sliding-window path, not FFN
  or norms.
- The method is accessible from the `ShramForCausalLM` public interface.
- The count is stable: two calls on the same model return the same value regardless of forward
  pass state.

**Tests:**
- **Scaling**: a model with 2× `num_hidden_layers` returns exactly 2× the MoSRAH parameter count.
  Verifies the aggregation logic without hardcoding any concrete number.
- **Partition**: MoSRAH count is strictly less than the total model parameter count — it is a
  proper subset.
- **Stability**: two calls on the same model return the same value.

**Preliminary strategy:** passthrough — `MoSRAHLayer` exposes its own count, `DecoderLayer`
aggregates, `ShramModel` sums across layers, `ShramForCausalLM` delegates. Method name to be
confirmed with user before implementation.

---

### Unit 19.E (Blocker) — E2E torch-dynamo compile coverage gap

**Responsibility:** Add torch dynamo compile verification to the end-to-end test suite so that the existing compile failure is surfaced as a test failure rather than being silently missed.

**Context of Correctness:**

Upon attempt to use this system in the real world, it was identified torch dynamo is not operational. However, no test is raising. This situation should not be possible, as the end to end test series should be testing torch dynamo as well. An oversight means it is not. We must ensure the tests correctly fail before debugging is possible.

**Invariants this unit must satisfy:**

- The end-to-end test suite contains a test that calls `torch.compile` on the model and executes a forward pass through the compiled result.
- That test fails on the current implementation.
- No currently passing tests are broken by this addition.

**Tests:**

- Call `torch.compile(model)` and execute a forward pass. Verify this test fails on the current codebase — failure here is the definition of done for this unit.

**Preliminary implementation strategy:**

- Locate the existing end-to-end test file and add the compile test alongside the existing smoke tests, reusing the same config and input setup already established there.
- Do not mark this unit complete if the test passes — that contradicts the known failure and must be investigated before proceeding.

---

### Unit 19.F (Blocker) — SlidingWindowAttention torch.compile failure

**Responsibility:** Fix `_make_block_mask` in `SlidingWindowAttention` to eliminate the data-dependent control flow that prevents torch dynamo from compiling the flex attention block mask.

**Context of Correctness:**

The sliding window attention system takes certain liberties which are incompatible with torch dynamo. In specific, it mixes control flow with data flow in a manner that is incompatible with flex attention's desire to build flex attention windows. In the sliding window attention formulation a cumsum is used to know the positions of semantic tokens given the elements of the attention mask. This, however, means that when dynamo builds a block attention mask it is attempting to compile control flow on a data-dependent quantity.

**Invariants this unit must satisfy:**

- `SlidingWindowAttention` compiles successfully under `torch.compile`.
- The compiled model produces identical attention outputs to the uncompiled model for all valid inputs.
- Active mask structures that are not supported by the revised implementation raise explicitly rather than silently producing incorrect results.
- All existing `SlidingWindowAttention` tests continue to pass.

**Tests:**

- Verify `torch.compile` on a model containing `SlidingWindowAttention` followed by a forward pass completes without error.
- Verify compiled and uncompiled forward passes produce numerically identical outputs on the same inputs.
- Verify unsupported mask structures raise rather than producing wrong results silently.

**Preliminary implementation strategy:**

- **Research required before implementation:** Verify exactly what structures `active_mask` can hold in practice across all call sites (training, cached inference, masked continuation). This determines which structures must be supported and which may raise.
- If `active_mask` is always left-justified (all `True` values precede all `False` values): assert left-justification at the `_make_block_mask` boundary and raise on violation; replace the `cumsum`-based semantic position computation with `torch.arange`, which is data-independent and dynamo-safe.
- Verify the `arange`-based positions produce identical masking behavior to the `cumsum` approach for all valid inputs before marking this unit complete.
- If the research reveals mask structures other than left-justified are valid, surface them explicitly before proceeding — do not resolve the gap autonomously.

---

### Unit 19.F.1 — Position-zero constraint: compile-compatible enforcement (revised)

**Responsibility:** Enforce the uncached starting-position constraint in both compiled and eager execution modes without graph breaks.

**Context of Correctness:**

Error messages at misuse boundaries are a long-term support requirement for any system whose outputs must be trusted. One such error condition is nonzero starting positions passed to an uncached forward — there is no prior KV state to justify positions other than zero, and the violation produces silently incorrect RoPE encoding and attention outputs. The check belongs in `huggingface.py`, before the main model runs, so it intercepts the misuse at the outermost boundary.

In compiled mode, data-dependent Python control flow including `raise` causes graph breaks. The sole exception is `torch._check` used with `capture_scalar_outputs = True`: this configuration enables `.item()` to be captured by dynamo as a SymInt, allowing the assertion to fold into the compiled graph without breaking it. However when `torch._check` fires, the error surfaces at the C++ level — the message string is not cleanly propagated; the call stack lands on the calling function, and only that frame is reported.

Therefore the most effective approach is to assume `capture_scalar_outputs = True` and implement an eager/compiled branch split inside a dedicated helper, triggered via `torch._check` on invalid conditions. The helper's docstring becomes the primary diagnostic surface.

**Invariants this unit must satisfy:**

- Uncached forward with nonzero starting positions raises in both compiled and eager execution
- The error is sufficient for the caller to identify the misuse and correct it

**Tests:**

- Compile the validation method in isolation; verify it raises under a violating condition
- Verify it raises under the same condition in eager mode

**Audit:**

- That the call site passes the correct condition tensor to the validator

**Preliminary implementation strategy:**

- `capture_scalar_outputs = True` is a precondition; 19.F.2 owns the warning when it is absent
- A dedicated static method owns the validation, branching on `torch.compiler.is_compiling()`: compiled path uses `.item()` + `torch._check`; eager path uses `.item()` + direct raise
- The method is static so it can be compiled and tested in isolation
- The method's docstring specifies what a failure means and how to resolve it
- The bool tensor condition is computed at the call site and passed into the method

---

### Unit 19.F.2 — Compile-time warning for missing `capture_scalar_outputs`

**Responsibility:** Raise an error at compile time when `capture_scalar_outputs = True` is absent, making the missing precondition immediately visible rather than leaving the user to discover it from downstream behaviour.

**Context of Correctness:**

`capture_scalar_outputs = True` is a non-default dynamo flag required for the safety checks established in 19.F.1 to fold into compiled graphs. Without it those checks silently revert to graph-breaking behaviour or are skipped. A compiled model with fewer safety properties than its eager counterpart produces no diagnostic output to that effect — the user has no indication that the protections they observed in eager mode are absent. For a research baseline this is a support liability: misconfiguration is invisible until something goes wrong downstream.

`torch.compiler.is_compiling()` returns True during dynamo tracing. `torch._dynamo.config.capture_scalar_outputs` is readable as a Python attribute at trace time. Together these make it possible to detect the misconfiguration at the exact moment it takes effect and raise before the compiled region executes.

**Invariants this unit must satisfy:**

- When the model is compiled without `capture_scalar_outputs = True`, an error is raised that identifies the missing flag and its consequence
- When `capture_scalar_outputs = True` is set, no error fires

**Tests:**

- Compile a minimal stand-in containing the check without the flag set; verify the error fires
- Verify no error fires when the flag is set

**Preliminary implementation strategy:**

- A dedicated static method raises the error, gated on `torch.compiler.is_compiling()` and the flag value
- Called at the entry point of `ShramForCausalLM.forward` so it fires on the first compilation pass
- Error message names the flag, its required value, and the consequence of absence

---

### Unit 19.G (Blocker) — RotaryEmbedding: preallocate to maximum_sequence_length, eliminate .item()

**Responsibility:** Rebuild `RotaryEmbedding` to preallocate the cos/sin table to a fixed `maximum_sequence_length` at construction time, eliminating the `position_ids.max().item()` call in `forward` that causes a graph break under torch.compile.

**Context of Correctness:**

The current lazy extension design calls `position_ids.max().item()` in every forward pass to determine the required cache length. This forces a CPU sync (data-dependent scalar extraction) and causes a dynamo graph break. Since the system is moving toward fixed-length preallocation for inference (Unit 19.H), the correct resolution is to preallocate at construction to a caller-supplied `maximum_sequence_length`. The `forward` path then only performs metadata checks (dtype, device) to determine if a rebuild is needed — both of which dynamo can handle without a break.

**Invariants this unit must satisfy:**

- `RotaryEmbedding` accepts a new required `maximum_sequence_length: int` constructor parameter.
- The cos/sin table is built at construction time to cover positions `[0, maximum_sequence_length)`. No lazy extension occurs on the first forward call. No remaining initial sequence length.
- `forward` contains no `.item()` call. Cache validity is checked via dtype and device metadata only.
- All existing `RotaryEmbedding` tests continue to pass. Constructor call sites are updated to supply `maximum_sequence_length`.
- For `mode="yarn"`, logic is changed as appropriate. 

**Tests:**

- Existing rope tests updated for new constructor signature — all must pass.
- Verify no `.item()` call appears in `forward` after the change.

**Audit**

- Verify by inspection no further rope graph breaks occur. 
---

### Unit 19.G.0 (Blocker) — Expert packing: interface consolidation

**Responsibility:** Refactor the expert packing interface to group the setup payload and entry tensors as structured inputs, and eliminate repeated identical gather-scatter operations.

**Context of Correctness:**

Code quality is an unconditional requirement in this project, as stated in job.md. Sane API signatures are part of that requirement. `pack_experts` currently takes seven parameters; the setup payload — three values always produced and forwarded together — is disaggregated into individual arguments, and the three entry tensors — which all undergo the same gather-scatter operation — are treated as separate concerns. This was marginally acceptable at its current size. Unit 19.G.1 must add at least one further parameter. That addition crosses the line from questionable into a clear violation of the code quality standard.

A static analysis of the call sites confirms that the setup payload always travels as a unit and is never partially forwarded. The entry tensors always undergo structurally identical operations. Both groupings are natural; the current interface artificially disaggregates them. Given the code quality axiom and the necessity of 19.G.1's additions, this refactor is required before 19.G.1 can proceed without producing code that fails job.md's unconditional standard.

**Invariants this unit must satisfy:**

- `setup_packing` returns a single auxiliary payload; callers forward it whole to `pack_experts` and `unpack_experts`
- `pack_experts` accepts entry tensors as a mapping from string keys to tensors, the setup payload, `selected_heads`, and `num_experts`; it returns a mapping from the same string keys to their packed counterparts, plus `unpacking_mask` as a separate output
- `unpack_experts` accepts `expert_outputs`, the setup payload, `unpacking_mask`, and `selected_heads`; its return type and shape contract are unchanged
- For all valid inputs, the packed tensor values produced by `pack_experts` are numerically identical to those produced by the previous implementation
- For all valid inputs, `unpack_experts` produces numerically identical output to the previous implementation
- No new behavior is introduced; this unit changes structure only

**Tests:**

- All existing tests in `test_expert_packing.py` pass with call sites updated to the new interface; no assertion values change — behavioral equivalence is the sole standard of correctness for this unit

**Preliminary implementation strategy:**

- `setup_packing` returns a dict; `pack_experts` and `unpack_experts` extract what they need from it by key
- Inside `pack_experts`, a single loop over the entries dict allocates output buffers and performs gather-scatter, using each tensor's trailing shape to determine index expansion — no helper function needed
- The data-dependent `max_tokens_per_expert` computation remains; 19.G.1 replaces it with `packed_length`
- Call site in `mosrah.py` updated to use the new interface

---

### Unit 19.G.1 (Blocker) — Expert packing: static T preallocation and overflow detection

**Responsibility:** Replace the data-dependent packed time dimension T with a statically derived allocation, and introduce compile-compatible overflow detection.

**Context of Correctness:**

Expert packing currently sizes the packed time dimension T by computing the maximum per-head token count at runtime — a data-dependent shape that dynamo cannot trace without breaking the graph. The expected token count per head is fully derivable from config: `training_sequence_length * num_selected_heads / num_mosrah_heads` is the average under perfectly balanced routing, a compile-time constant. Overallocating this by a configurable factor provides margin for routing imbalance while keeping T static.

This unit is also foundational for the static cache rebuild in 19.G.4, which requires static MoSRAH buffer sizes. The same T derived here is the natural size for those buffers — establishing it now avoids a redundant derivation later and ensures the two subsystems agree on the same value.

Overflow — actual per-head count exceeding T — is now detectable via `torch._check` with `capture_scalar_outputs = True` (established in 19.F.1 and 19.F.2), which fires in both compiled and eager modes. The overallocation factor is an architectural parameter governing core data structure sizes and belongs in `ShramConfig`. The load balancing system provides the ongoing guarantee that actual counts stay within T under normal operation; 19.G.3 addresses the correctness of that guarantee.

**Invariants this unit must satisfy:**

- `ShramConfig` declares `mosrah_overallocation_factor: float` with value > 1.0, default 1.1, which survives `to_dict`/`from_dict` roundtrip
- `ShramConfig` exposes `mosrah_packed_length` as a computed property equal to `ceil(training_sequence_length * num_selected_heads / num_mosrah_heads * mosrah_overallocation_factor)` — the single source of truth for the packed time dimension consumed by all downstream users
- When actual per-head token count exceeds `mosrah_packed_length`, the system raises with a message identifying `mosrah_overallocation_factor` as the remedy, in both compiled and eager modes

**Tests:**

- Verify `mosrah_overallocation_factor` roundtrips through `to_dict`/`from_dict`
- Verify values ≤ 1.0 raise `ValueError` at construction
- Verify `mosrah_packed_length` returns the correct value for any valid config
- Verify overflow raises with an informative message in eager mode
- Compile the overflow check in isolation; verify it raises in compiled mode

**Audit:**

- That no data-dependent `.item()` calls remain in the T-sizing path after this unit

**Preliminary implementation strategy:**

- Add `mosrah_overallocation_factor` to `ShramConfig` with validation (> 1.0) and default 1.1
- Add `mosrah_packed_length` as a computed property on `ShramConfig`; all consumers read T from there
- Derive T in `pack_experts` from config parameters directly; remove `max_tokens_per_expert = int(tokens_per_expert.max().item())`
- Overflow detection follows the dedicated static method pattern from 19.F.1: a static method with a precise docstring, branching on `torch.compiler.is_compiling()`
- `_bincount_rows` remains for count computation in this unit; that is addressed in 19.G.2

---

### Unit 19.G.2 (Blocker) — Expert packing: replace `_bincount_rows` with `scatter_add`

**Responsibility:** Replace the `_bincount_rows` helper with a `scatter_add` into a pre-sized buffer, eliminating the dynamic output shape that causes a graph break in the count computation path.

**Context of Correctness:**

Computing per-expert token counts is a prerequisite for constructing the unpacking mask and verifying occupancy against `mosrah_packed_length`. The current implementation uses `torch.bincount`, whose output shape depends on the maximum value in the input — a data-dependent shape that dynamo cannot trace without a graph break regardless of the `minlength` argument.

The per-expert count tensor has shape `(B, num_mosrah_heads)` — fully known at compile time. `scatter_add` into a pre-sized zero buffer of that shape accumulates ones at the expert index positions to produce identical counts with a static output shape. No alternative achieves this without either dynamic shapes or sequential iteration over batch items.

**Invariants this unit must satisfy:**

- Per-expert token counts are produced into a pre-sized `(B, num_mosrah_heads)` buffer — no dynamic output shapes
- The counts are numerically identical to those previously produced by `_bincount_rows`
- `_bincount_rows` is removed

**Tests:**

- Verify counts match `_bincount_rows` output for a range of routing configurations
- Verify no graph break occurs in the count computation path when compiled

**Audit:**

- That no `bincount` calls remain anywhere in the packing path
- Verify functional equivalence of the new count computation against the old `_bincount_rows` approach — either by manual inspection of the algebra or by running both on the same inputs and comparing outputs before removal

**Preliminary implementation strategy:**

- Allocate a zeros buffer of shape `(B, num_mosrah_heads)` on the correct device
- Scatter ones at `flattened_selected_heads` positions via `scatter_add`
- Remove `_bincount_rows` entirely

---

### Unit 19.G.3 (Blocker) — Load balance frequency aggregation: p-mean and `load_balance_p`

**Responsibility:** Replace arithmetic mean routing frequency aggregation with p-mean, making the load balance correction signal sensitive to per-item allocation spikes that cause overflow.

**Context of Correctness:**

Load balancing exists to keep per-head token counts within `mosrah_packed_length`. Unlike MoE routing, MoSRAH routing is fully data-dependent — a head may be rarely selected but receive a large number of tokens when it is, causing overflow regardless of its average allocation. `expert_bias` is a global parameter with L degrees of freedom applied equally to all inputs; it cannot protect individual batch items directly. What it can do is shift the statistical tendency of each head's allocation. The relevant question is therefore not what the mean allocation is, but what aggregation of per-item frequencies best uses those L degrees of freedom to minimize overflow probability.

Arithmetic mean is blind to per-item spikes: a head that averages near 1/L while occasionally spiking produces no correction signal. P-mean with p > 1 penalizes distributional spread — a head with high per-item variance inflates its p-mean above 1/L even when its arithmetic mean is near 1/L, triggering correction. At p=2 this is the RMS, which directly captures the combination of mean and variance. Higher p increases outlier sensitivity; p=1 reduces to the original arithmetic mean. The correct p is a research question, so it is configurable.

**Invariants this unit must satisfy:**

- `ShramConfig` declares `load_balance_p: float`, default 2.0, which survives `to_dict`/`from_dict` roundtrip
- `routing_freqs` passed to `LoadBalanceLoss` is shape `(L,)`, computed as the p-mean over batch items of per-item frequencies
- `MaxVio` follows the paper's formula L · max_l(f_l − 1/L) applied to `routing_freqs`; it benefits from the improved signal because `routing_freqs` is now p-mean rather than arithmetic mean (literature compatible)

**Tests:**

- Verify `load_balance_p` roundtrips through `to_dict`/`from_dict`
- Verify a batch where one item spikes on a head produces a higher `routing_freqs` value for that head than arithmetic mean alone would give

**Preliminary implementation strategy:**

- Add `load_balance_p` to `ShramConfig` with default 2.0
- Compute per-item frequencies `(B, L)` — sum active assignments over sequence per item, normalize per item
- Apply p-mean over batch dimension: `(mean_b(f_{b,l}^p))^(1/p)` → `(L,)` passed to `LoadBalanceLoss` unchanged
- `LoadBalanceLoss` requires no changes

---

### Unit 19.G.4 (Blocker) — Restore mask symmetry for compiled inference via create_masks_for_generate override

**Responsibility:** Override `create_masks_for_generate` on `ShramForCausalLM` to return the 2D `attention_mask` unchanged, and restore `is_compileable = True` on `ShramLayerCache` and `ShramCache`.

**Context of Correctness:**

HuggingFace's `prepare_inputs_for_generation` in `GenerationMixin` behaves differently depending on whether a cache entry is marked as compileable. When `is_compileable = True`, it calls `create_masks_for_generate`, which converts the 2D boolean attention mask into a 4D causal additive-bias mask before passing it downstream. This transformation exists to accelerate traditional torch transformer instances under compilation by precomputing the causal mask once — and it is the correct choice for models that consume that format natively.

SHRAM, however, uses flex attention with custom masking; flex attention works with boolean masks instead, has been implemented internally with causality, and only depends on the external mask to know the tokens which are not dead. The existing system expects a 2D boolean mask, and relies on it to ignore attention between dead tokens during training. As such the mask is incompatible, as currently specified, with the downstream system. This must be resolved.

Given the usage of flex attention, the performance implications that motivated the usage of an additive mask do not actually exist. This means the most straightforward option is to standardize back into a 2D mask form; the reason for the additive mask is not relevant. In theory, it would be possible to reverse the additive mask back into a boolean mask by treating sufficiently large negative logits as dead-token signals. In practice, this wastes additional computation for no reason. Instead, HuggingFace provides the `create_masks_for_generate` override point precisely so model implementations can specify their own custom behavior when going through compiled transforms. Overriding this such that it passes the mask through unchanged is the optimal option, as the computation is skipped and it uses the supported framework hooks for the process.

**Invariants this unit must satisfy:**

- `ShramForCausalLM.create_masks_for_generate` exists and returns the 2D `attention_mask` argument unchanged
- Compiled and non-compiled inference pathways present the same 2D boolean mask format to the SHRAM attention stack
- `ShramLayerCache.is_compileable = True`
- `ShramCache.is_compileable = True`
- All existing inference tests continue to pass

**Tests:**

- Verify `create_masks_for_generate` returns its `attention_mask` argument unchanged
- Verify tests previously failing due to this asymmetry now pass

**Preliminary implementation strategy:**

- Override `create_masks_for_generate` on `ShramForCausalLM`; inspect the HuggingFace signature and return the `attention_mask` unchanged with no other modifications
- Set `is_compileable = True` on `ShramLayerCache`; verify `ShramCache` carries this flag or update if not
- If the HuggingFace override contract proves to require more than a passthrough, surface it rather than filling the gap silently

---

### Unit 19.G.5 (Blocker) — Static cache rebuild for compiled inference

**Responsibility:** Rebuild `MoSRAHCache` and `LocalSlidingWindowLayerCache` to satisfy the HuggingFace static cache contract, eliminating all graph breaks in the cache update path and enabling end-to-end compiled inference.

**Context of Correctness:**

Given the time investment in this system and its intended use across multiple research efforts, the model should support compiled inference.

Compiled inference requires all cache operations to execute within the compiled graph on statically-shaped tensors. `MoSRAHCache` violates this in three independently graph-breaking ways: dynamic buffer allocation on overflow (`_expand()`), data-dependent overflow detection via `.item()`, and data-dependent-length index extraction via `torch.where`. `LocalSlidingWindowLayerCache` is already structurally static — its buffer is fixed at construction and its update path introduces no dynamic shapes. The maximum per-`(batch, head)` slot occupancy over a full inference run is `ceil(inference_sequence_length * K / L * mosrah_overallocation_factor)` — fully derivable from config, making static pre-allocation of `MoSRAHCache` possible and establishing `mosrah_cache_length` as the required config property. HuggingFace's generation machinery routes between static and dynamic cache execution via `is_compileable` and `get_max_cache_shape()`; without truthful declarations on both caches, the compiled path is unavailable regardless of how cache internals are structured.

**Invariants this unit must satisfy:**

- `ShramConfig` exposes `mosrah_cache_length` as a computed property equal to `ceil(inference_sequence_length * num_selected_heads / num_mosrah_heads * mosrah_overallocation_factor)`, which survives `to_dict`/`from_dict` roundtrip
- `MoSRAHCache` buffer capacity is fixed at `mosrah_cache_length` at construction and never changes during inference
- `MoSRAHCache.update()` contains no data-dependent shape operations
- When any `(batch, head)` slot would exceed `mosrah_cache_length`, the system raises in both compiled and eager modes
- `MoSRAHCache.is_compileable = True` and `get_max_cache_shape()` returns `mosrah_cache_length`
- `LocalSlidingWindowLayerCache.is_compileable = True` and `get_max_cache_shape()` returns `sliding_window`
- `ShramLayerCache` and `ShramCache` construct from `ShramConfig` rather than individual parameters
- `ShramLayerCache.is_compileable = True` and `get_max_cache_shape()` returns `inference_sequence_length`
- `ShramCache.is_compileable = True` and `max_cache_len` delegates to `layers[0].get_max_cache_shape()`
- `SlowMoSRAHCache` accepts `mosrah_cache_length` as its static capacity to maintain oracle contract with `MoSRAHCache`
- A compiled model with `ShramCache` executes cached inference without graph breaks

**Tests:**

- Verify `mosrah_cache_length` returns the correct value for a given config
- Verify `mosrah_cache_length` survives `to_dict`/`from_dict` roundtrip
- Compile `MoSRAHCache.update()` and run over several simulated decode steps; verify no graph breaks
- Compile `LocalSlidingWindowLayerCache.update()` and run over several simulated decode steps; verify no graph breaks
- Verify `ShramLayerCache.get_max_cache_shape()` returns `inference_sequence_length`
- Verify `ShramCache.max_cache_len` returns `inference_sequence_length`
- Verify overflow raises in eager mode
- Verify overflow raises in compiled mode
- Verify end-to-end `generate()` with a compiled model completes without error

**Audit:**

- No `_expand()` or equivalent dynamic reallocation remains in `MoSRAHCache`
- No `.item()` call remains in the `MoSRAHCache.update()` hot path
- No `torch.where` or equivalent variable-shape index extraction remains in `MoSRAHCache.update()`
- `get_max_cache_shape()` returns correct values on `MoSRAHCache`, `LocalSlidingWindowLayerCache`, and `ShramLayerCache`
- `is_compileable = True` on `MoSRAHCache`, `LocalSlidingWindowLayerCache`, `ShramLayerCache`, and `ShramCache`

**Preliminary implementation strategy:**

- Add `mosrah_cache_length` as a computed property to `ShramConfig`: `ceil(inference_sequence_length * num_selected_heads / num_mosrah_heads * mosrah_overallocation_factor)`; all consumers of the MoSRAH cache buffer size read from there
- `MoSRAHCache`: pre-allocate `keys` and `values` to `mosrah_cache_length` at construction; remove `_expand()`; update constructor to accept `mosrah_cache_length` in place of `initial_buffer_size`
- Replace the `.item()` overflow guard in `update()` with a dedicated `torch._check` helper following the pattern established in 19.F.1
- Replace `torch.where(active_mask)` scatter with a fixed-shape transfer matrix operation; confirmed graph-break-free by probe
- `SlowMoSRAHCache`: update constructor to accept `mosrah_cache_length` directly (not full config — it is a test oracle, not a model component); pre-allocate to that capacity so oracle comparison with `MoSRAHCache` remains valid
- Refactor `ShramLayerCache` and `ShramCache` constructors to accept `(config: ShramConfig, batch_size, device)` rather than individual parameters; implement `get_max_cache_shape()` on `ShramLayerCache` returning `config.inference_sequence_length`; implement `max_cache_len` on `ShramCache` delegating to `layers[0].get_max_cache_shape()`
- Set `is_compileable = True` on `MoSRAHCache`, `LocalSlidingWindowLayerCache`, `ShramLayerCache`, and `ShramCache`
- Add compiled inference coverage to the e2e test suite

---

### Unit 20.A (Blocker) — Release pipeline: dev repository staging and E2E test gate

**Responsibility:** Restructure the release pipeline so that publication is gated on E2E tests passing against a staging Hub repository, and configure the test suite to exclude E2E tests from normal runs while making them target-configurable.

**Context of Correctness:**

Attempts to repair an issue has revealed a publication deadlock. In order for publication of new versions to happen tests must pass. However, certain end-to-end tests will fail until a new version is published. This forms a deadlock that must be resolved while continuing to optimize for correctness.

For the end-to-end tests to be run they must be pointed at a huggingface repository from which to load; we now know there are verifiable differences between remote and local repository loading systems. However, the repositories do not work well with config-only models; such models cannot be versioned or put in separate folders. As such, any upload is inherently destructive without rollback.

While it would in theory be possible to use the git nature of huggingface model repositories to roll back any damage automatically, alternatives that keep test artifacts out of the production space are preferable. As such, the best case solution is the integration of test repositories into the testing stream, which can be uploaded to and tested against, before uploading to the production repositories themselves.

**Invariants:**
- A staging Hub repository exists for each model (llama3, shram) as an intermediate publication target, distinct from the production repository
- The release pipeline publishes to the staging repository first, runs E2E network tests against it, and only publishes to the production repository if those tests pass
- The GitHub release draft is only promoted to published after production publication succeeds
- E2E network tests are excluded from the normal test suite by default via `addopts` in `pytest.ini`
- E2E network tests accept a `--hub` argument accepting `dev` or `main`, defaulting to `main`, which configures the target repository
- The production Hub repository is never written to unless E2E tests against the staging repository have passed
- A one-minute delay is introduced before post-upload tests run to allow Hub propagation

**Tests**
- Modify such that `pytest tests/shram/` with no flags does not execute any network-marked tests
- Modify such that `pytest tests/shram/ -m network --hub=dev` runs network tests against the staging directory
- Modify such that `pytest tests/shram/ -m network --hub=main` runs network tests against the production repository.

**Audit**
- Verify `pytest tests/shram/` with no flags does not execute any network-marked tests
- Verify `pytest tests/shram/ -m network --hub=dev` runs network tests against the staging repository
- Verify `pytest tests/shram/ -m network --hub=main` runs network tests against the production repository
- Workflow behavior is verified by running a release through the pipeline and observing job sequencing

**Preliminary Implementation Strategy:**
- Create staging Hub repositories: `smithblack-0/SHRAM-dev` and `smithblack-0/LLAMA3-dev`
- Register `@pytest.mark.network` and add `addopts = -m "not network"` to `pytest.ini`
- Add `--hub` option to `conftest.py` via `pytest_addoption`; expose as a fixture replacing the hardcoded `HUB_REPO` constant in `test_end_to_end.py`
- Restructure `shram.yml` and `llama3.yml` to four jobs: `test` (no network marks), `upload_dev`, `post_test` (network, `--hub=dev`, one minute sleep), `upload_main` (conditional on post_test passing)
- Add `publish_release` job calling `gh release edit --draft=false` after `upload_main` succeeds
- Change orchestrator trigger from `release: types: [published]` to `release: types: [created]`; developers create releases as drafts

**Note:** During implementation, `cache_position` was found to no longer be passed by `GenerationMixin` in the current transformers version. Fixed `Llama3ForCausalLM.forward()` to derive position and mask from `past_key_values.get_seq_length()` instead. `cache_position` remains in the signature for contract compatibility.

---

### Unit 20.B — stage_for_hub.py: single-file inline staging

**Responsibility:** Replace the staging system with one that produces a single merged `huggingface.py` containing all model Python code, satisfying HuggingFace's single-model-file loading contract.

**Context of Correctness:**

Real world testing has revealed an issue. It is not in fact possible to load our model through AutoModel with a from_pretrained predicate on a local disk. Further investigation has lead to the understanding that while remote fetches recursively resolve all dependencies into the huggingface cache, local loads do not. As such, only the files that are directly imported in huggingface.py are being transferred into the cache, causing a crash when the system tries to load it. This is found in get_cached_module_file.

While this issue can, in theory, be corrected by directly registering the model, this bypasses the premise of isolation the test harness was designed under. No minor tweak of huggingface will fix the issue. Instead, flattening the model down to one file will fix the issue. We resolve the model to a single huggingface.py file instead. 

**Invariants:**
- `stage(source_dir, dest_dir)` stages by looking into source directory and moving a staged version into destination directory.
- The staging output contains exactly one Python model-containing file: `huggingface.py`
- That file contains all model Python code reachable from `source_dir/huggingface.py` via relative imports, inlined in dependency order
- No relative imports appear anywhere in the merged `huggingface.py`
- External imports appear exactly once; duplicate occurrences are commented out with `# `
- All docstrings and comments from source files are preserved verbatim in the merged output
- No source file under `source_dir` is modified by staging; all writes go to `dest_dir` only
- Other relevant files present in `source_dir` are copied into `dest_dir` unchanged
- `AutoModelForCausalLM.from_pretrained(path, local_files_only=True, trust_remote_code=True)` succeeds on a directory produced by `stage()`

**Tests:**
- Stage a known source directory; verify the output contains exactly one `.py` file named `huggingface.py`
- Verify no relative import statements remain in the merged file
- Verify `test_model_save_load_roundtrip` passes against a staged directory
- Verify non-Python files are present in the staging output unchanged
- Verify docstrings and comments from source files are present in the merged output

**Preliminary Implementation Strategy:**

Adapt the algorithm from [python-import-inliner](https://github.com/dobrakmato/python-import-inliner). Maintain `inlined` (set of already-processed file paths) and `seen_imports` (set of already-emitted external import lines). `inline_file(file, base, out)` iterates lines: relative imports are resolved and recursively inlined if not yet seen, otherwise commented out; external `import x` lines are emitted once then commented on recurrence; all other lines emit as-is. `stage()` calls `inline_file` starting from `source_dir / "huggingface.py"`, writes the result to `dest_dir / "huggingface.py"`, then copies non-Python files with `shutil.copy2`.

---

### Unit 21 

**Responsibility*

Fix capacity issues that are causing training overflows

**Context of Correctnesss**

Between huggingface and the current load balancing strategy there is a problem. Initialization and usage of models during training are currently regularly exceed the capacity of limits of the MoSRAH formulation; during initialization in particular capacity is not evenly distributed in a manner such that over double the capacity is sometimes needed. This should not have happened, as the initialization mechanism using xavior uniform should have maintained variance within an acceptable region

Further investigation has revealed huggingface manually overrides any initialization originally introduced, ensuring that the even capacity usage we were going to rely on does not in fact exist. Capacity limits are, bluntly, an issue again, as is the breakage of sane initialization assumptions and consequent possible training instability.

### Unit 21.A — huggingface initialization fix

**Responsibility**

Modify initialization to be robust against huggingface trickery to maintain variance assumptions used for sane routing assignment.

**Context of Correctness**

Originally, it was presumed variance would remain approximately the same as the input sequence due to the usage of xavier uniform initialization on linear layers. This would have allowed routing operations to be chosen with about the same variance regardless, and thus equally distribute routing. This assumption turns out to be wrong. 

Applying skip-init, however, smooths the underlying variance issues which may make training difficult. It should also stabilize training in a variety of useful ways.

**Invariants**

- Skip init with value of zero is applied at all residual connections on the primary layers
- Layers start with a trainable vector of zeros masking their residual contributions but allowing gradients to turn them on during training.

**Audit**

- Verify solution is installed. 

**Preliminary implementation strategy**

- It looks like decoder_layer.py is the attachment point.

### Unit 21.B - Training capacity fix.

**Responsibility**

Permanent enduring capacity fix by explicit capacity handling in the routing stage

**Context of Correctness**

While fixing stability is well and good, it does not guarentee permanent training or inference capacity. In theory, if routing could be modified such that it cannot assign tokens to full buckets - that is if it respected capacity limits - it would be possible to avoid having capacity overflows in the first place. This may cause minor differences between inference and training behavior, but htis is acceptable

**Invariants**

- Routing is modified to prioritize the tokens which most want a given head when over capacity
- Routing is modified such that tokens simply choose other best-case options when out of capacity.
- Routing and broader subsystems are modified in a way that makes testing possible

**Testing**

- Add test for the "balance_capacity" static helper method, verifying functionality, in the routing suite
- Confirm existing tests continue to pass

**Preliminary implementation strategy**

- The logits can be intercepted in their biased (B, N, L) form. Since logits are ordered the same as probabilities, this will also tell us what will be most strongly chosen later. 
- Masking out any logits to -1e8 if it would exceed capacity limits is sufficient to maintain load balancing. This could be trivially done if the value of the lowest ranked sorted logit at capacity was known; one would then simply mask out everything where threshold < logits.
- Finding threshold is straightforward when training. If training, capacity is never used as in inference, and always starts at zero. Taking a kthvalue, with k=capacity, will allow the extraction of the logits at capacity; note inversion is needed to get the largest element instead. However, inference will need a more complex pathway. 
- For inference, a "used_capacity" tensor of shape (B, L) or (B, 1, L) must be provided from the mosrah cache length.
- The used capacity inference pattern requires a bit of padding and a while statement to execute properly. When the gathered topk is shorter than the possible capacity, it should be padded with a 1e8 thresold "allow any" value, then any index greater than the length can be redirected into the padding statement with a torch.while statement
- It would make more sense to distinguish the pathway by passing in the used_capacity tensor, then using kthvalue or topk depending on if it is none or not. But remember, default values are not allowed at this level. 

**Closure note (Unit 21.B):**
The Sinkhorn alternating-projection implementation was foundationally broken. Convergence was too slow and too unreliable at the routing densities required for this architecture — not a tuning problem, a structural one. This unit is superseded by Unit 21.C.

---

### Unit 21.C — Bidding-based capacity enforcement

**Responsibility**

Replace `balance_capacity` with an implementation that simultaneously satisfies both the per-expert capacity bound and the per-token minimum-choice bound, and certify it as a verified unit. The Sinkhorn implementation in Unit 21.B could not reliably satisfy these constraints together and is superseded by this unit.

**Context of Correctness**

Routing correctness requires two constraints to hold simultaneously: every expert's token count must not exceed its capacity budget (column bound), and every token must retain at least K available expert choices (row bound). These are not independent — satisfying one greedily can violate the other. When a token is left with fewer than K unmasked experts, `topk(K)` fills the remaining slots via tie-breaking on effectively-zero masked entries, producing ghost selections that inflate actual head occupancy above capacity and cause an overflow crash in `pack_experts`.

The Sinkhorn approach (Unit 21.B) alternated row/column projections on a shifted logit space. In theory this should converge; in practice convergence was too slow and too unreliable. A deferred-acceptance (Gale-Shapley) bidding approach was developed in a standalone probe and validated empirically: the probe compiled successfully under `torch.compile(fullgraph=True)`, and 10 rounds covers convergence up to approximately the 98th percentile of routing densities — beyond 10 rounds you are in the top 2% of extreme-density cases not expected under normal training. This unit installs that algorithm to project standards.

**Invariants**

1. For every (batch, expert) pair: the number of non-masked token positions does not exceed `remaining_capacity` for that expert.
2. For every (batch, token) pair: the number of non-masked expert positions is at least K.
3. Invariants 1 and 2 hold simultaneously on every output of `balance_capacity`.
4. When called in training mode (no `used_capacity`) with N ≤ `capacity`, the output is identical to the input — no masking is applied.
5. When `used_capacity` is provided (inference mode), remaining capacity is computed per-expert as `capacity − used_capacity` clamped to zero; invariants 1 and 2 hold with respect to this per-expert budget.
6. When no valid joint assignment exists, the method raises `RuntimeError` in eager mode.
7. `max_bid_rounds` is a `ShramConfig` integer parameter, default 10, validated ≥ 1 at construction time; it is the iteration ceiling before invariant 6 fires.

**Tests**

- Joint constraint, training, tight budget: verify invariants 1 and 2 hold simultaneously when N approaches the column capacity budget.
- Joint constraint, inference, per-head remaining: provide `used_capacity` with mixed per-expert values; verify both invariants hold against the per-expert remaining budget.
- Fast-path, N ≤ capacity: verify output equals input unchanged.
- Non-convergence: infeasible config (total capacity < N * K) raises `RuntimeError` in eager mode.
- `max_bid_rounds` roundtrips through `to_dict`/`from_dict`.
- `max_bid_rounds = 0` raises `ValueError` at config construction.

**Preliminary implementation strategy**

The validated algorithm is deferred-acceptance (Gale-Shapley) bidding: tokens propose experts in preference order (proposals are monotone — never retracted); experts accept their top-capacity proposed tokens each round; the loop continues until all tokens have K accepted experts or `max_bid_rounds` is exhausted. Three single-pass fast paths precede the solver: training N ≤ capacity (return unchanged); row-topK precheck (if top-K selections already fit column budgets, mask and return); column-capacity precheck (if top-capacity selection already satisfies row minimum, mask and return). The bidding loop uses `torch.while_loop` for compile compatibility. Convergence checking follows the 19.F.1 pattern: `torch._check` in compiled mode, direct `RuntimeError` in eager. `max_bid_rounds = 10` covers convergence at approximately the 98th percentile of routing densities; beyond 10 rounds you are in the top 2% of extreme-density cases not expected under normal training. This should be documented as such. If the inference path's `remaining_capacity` as a (B, L) tensor is incompatible with the while_loop structure, surface and stop rather than adapting silently.

---

## Unit 22 - Cleanup 

The online version in huggingface is corrupt. There are also minor quality of life issues that need to be fixed

### Unit 22.A

**Responsibility**: 

Replace the existing assertion hack with the proper testing hook to remove dependency on capture_scalar_output. 

**Context of Correctness**

When the synchronous inline assertion system was orginally installed, it was found the only way to make it work was to use a dynamo capture_scalar_outputs flag, This resulted in the installation of three horribly slow but necessary tests. This involved forcing the capture of scalar outputs, and inserting tests at the following locations, and a test to ensure capture scalar output was set in the environment. 

  - src/shram/model/huggingface.py — two sites: position-zero check (~line 387) and _enforce_capture_scalar_outputs (~line 413)                           - src/shram/model/cache/mosrah_cache.py — overflow check (~line 357)                         
  - src/shram/model/attention/expert_packing.py — overflow check (~line 311)  

It is now known that torch._assert_async can accomplish the same thing without causing a graph break and in a manner that can be compiled. Given how slow capture_scalar_output is, it should be substituted, and shutting off the flag investigated.

**Invariants**

- The three test files listed above should have their checks replaced with torch._assert_async statements. Note messages cannot be longer than 256 characters.
- To prevent mass testing failure, only the compiled branch should be replaced like this.
- Validation is refactored to be sane; passes in parameters that are checked in the function. 

**Tests**
-  torch.AcceleratorError, RunTimeError, and AssertionError should be caught when checking for if the change worked.

**Preliminary implementation strategy**

- Most validation functions currently accept just a tensor bool. The logic to produce that bool should be moved inside the tests
- torch._assert_async accepts that tensor bool and a message no longer than 256 characters

### Unit 22.B — Remove `capture_scalar_outputs` dependency

**Responsibility:** Certify that `torch._assert_async`-based safety checks operate correctly under `torch.compile` without `capture_scalar_outputs`, and remove the enforcement infrastructure whose precondition has ceased to hold.

**Context of Correctness:**

This project's inline safety checks (capacity overflow, position-zero constraint) exist to catch invariant violations at compiled runtime. Their only value is that they fire. A check that is silently absent in the compiled model is worse than no check — it provides false confidence.

Unit 22.A replaced the `torch._check + .item()` mechanism with `torch._assert_async`, which operates on tensor booleans and requires no scalar extraction. That eliminated the dependency on `capture_scalar_outputs`. The enforcement guard installed in `ShramForCausalLM` was correct when the dependency existed: it detected misconfiguration at trace time before the checks could go silent. Now that the dependency is gone, the guard enforces a requirement that has ceased to exist.

A vacuous enforcement check is not neutral. It misleads future maintainers into believing a real dependency remains, imposes a compilation cost (dynamo captures every `.item()` as a SymInt when the flag is set), and embeds a false statement into the test contracts. Leaving it in place is a correctness violation of the documentation and test layer, even if the runtime behavior happens to be unaffected.

The alternative — leaving the flag in place — is less correct because the cost is real and the justification has evaporated.

**Invariants:**

1. Safety checks in the compiled model (capacity overflow, position-zero constraint) fire correctly with `capture_scalar_outputs` at its default value.
2. The compiled model produces no errors and no degraded behavior when `capture_scalar_outputs` is not set.
3. No test file sets `capture_scalar_outputs` as a prerequisite for any test to pass.

**Tests:**

All currently-passing compiled tests serve as the regression gate: they must pass without `capture_scalar_outputs = True` in setup. The test for the enforcement guard is deleted — it tested a method whose precondition no longer holds, and a test with a false premise cannot enforce a true invariant.

**Preliminary implementation avenues (not invariants):**

- Remove the enforcement method and its call site from `ShramForCausalLM.forward`.
- Remove save/set/restore boilerplate from all test files. Delete `TestCaptureScalarOutputsEnforcement` entirely.
- Affected files: `huggingface.py`, `test_end_to_end.py`, `test_expert_packing.py`, `test_huggingface.py`, `test_router.py`, `test_mosrah_cache.py`, `test_sliding_window_cache.py`.

### Unit 22.C — Expert packing: fixed-shape compact-to-padded transfer

**Responsibility**

Fix graph breaks in the compiled backward pass by replacing the boolean mask-based scatter/gather in `pack_experts` and `unpack_experts` with a fixed-shape integer index, without changing the external semantic contract of packing or unpacking.

**Context of Correctness**

Forward passes through the compiled model currently succeed. Backward passes do not: the boolean mask assignment used to transfer the compact expert-major stream into the padded tensor exposes a data-dependent shape to the compiler. Even though the system guarantees exactly B×N×K selected entries by invariant, the compiler sees the selected dimension as depending on mask contents and cannot trace through it cleanly in the backward pass.

The correct fix is a coordinate-space transformation: compute an integer index that maps each compact-stream position to its padded destination, then use scatter and gather on that index. The index values depend on routing, but the index shape is fixed by B, N, K, num_experts, and packed_len2gth — all statically known at trace time. This is the distinction the compiler needs.

Keeping the boolean mask as the transfer mechanism and adding assertions or static-shape hints was rejected. Such workarounds carry no semantic guarantee to the compiler and do not address the root cause.

**Invariants**

- For all valid inputs, `pack_experts` produces numerically identical output to the previous implementation.
- For all valid inputs, `unpack_experts` recovers the same token-choice output as the previous implementation.
- No tensor shape in the packing or unpacking transfer path depends on the number of True entries in any mask.
- The active mask is preserved in the packing output and remains available to downstream attention consumers.
- Padding positions in the packed tensor contain the padding fill value.
- Every routed copy is stored exactly once in the packed tensor.
- Inserted additional code is still done to standards; block comments and algorithm overviews are used liberally to ensure why execution occurs this way is clear, nad how execution occurs. 

**Tests**

- Behavioral equivalence: over a range of routing configurations including imbalanced routing, new packing and unpacking produce numerically identical results to the previous implementation.
- Round-trip: pack then unpack recovers the original compact expert-major stream exactly.
- Padding invariant: positions beyond each expert's token count contain the padding fill value after packing.
- Compile backward: `test_compile_uncached_backward` in `TestIntegrationCompilable` passes — this is the primary regression gate for this unit.

**Preliminary implementation strategy**

The proposed approach computes a flat integer destination index from `tokens_per_expert`. Given counts of shape `(B, L)` and packed length T:

```
block_counts   = tokens_per_expert.flatten()                    # (B*L,)
skipped        = T - block_counts                               # (B*L,)
skipped_before = exclusive_cumsum(skipped)                      # (B*L,)
skipped_stream = repeat_interleave(
                     skipped_before, block_counts,
                     output_size=B*N*K,
                 )                                              # (B*N*K,)
linear_destination = arange(B*N*K) + skipped_stream             # (B*N*K,)
```

The `output_size` parameter on `repeat_interleave` is what fixes the output shape at B×N×K regardless of routing distribution. Whether this form compiles cleanly under dynamo is an empirical question — if it does not, surface and stop rather than working around it silently.

`linear_destination` should be stored in the packing setup payload so unpacking reuses the same index without recomputation.




### Unit 23.A — Compiled inference-style execution: eval, no_grad, mixed-precision coverage

**Responsibility:** Add a dedicated test class that exercises the compiled model under inference-style execution contexts — eval mode, no_grad, and fp16 autocast — where an observed production Inductor failure occurs and the current suite provides no coverage.

**Context of Correctness:**

`TestIntegrationCompilable` covers compilation under training-style contexts (train+grad uncached forward, compiled generate). A production failure was observed under eval+no_grad+fp16 autocast: `InductorError: LoweringException: AssertionError: convert FlexibleLayout to FixedLayout first` at the FlexAttention boundary. These three contexts are qualitatively distinct from what is currently tested. In training mode with gradients, Inductor's fusion and layout finalization may incidentally produce FixedLayout tensors before reaching FlexAttention. In eval+no_grad, with no autograd graph to guide fusion, tensors produced by view+transpose+RoPE may remain FlexibleLayout all the way to FlexAttention lowering, triggering the assertion. Under fp16 autocast, dtype casting changes the operation sequence Inductor sees, potentially altering layout finalization order. None of these contexts are covered. A compile test suite that cannot detect a production Inductor failure at a known code site is not providing the coverage it claims.

**Invariants:**

1. A test class exists whose sole responsibility is the compiled model under inference-style execution contexts (eval, no_grad, mixed precision).
2. The class contains a test that compiles the model, runs it in eval+no_grad mode with labels and `use_cache=False`, and completes without error.
3. The class contains a test that compiles the model, runs it in eval+no_grad mode under fp16 autocast with labels and `use_cache=False`, and completes without error.
4. The class contains a test verifying compiled eval output matches eager eval output on identical inputs.
5. All tests use batch ≥ 2 and sequence length ≥ 128, with at least one attention_mask row shorter than the sequence dimension, exercising mixed-padding code paths.
6. All tests use the `device` fixture (CUDA skipif), consistent with the existing suite.

**Tests that certify them:**

- `test_compiled_eval_labeled_forward` — compiled eval+no_grad, batch=4, seq=128, mixed-length attention_mask, labels, `use_cache=False`
- `test_compiled_eval_autocast_labeled_forward` — same under `torch.amp.autocast("cuda", dtype=torch.float16)`
- `test_compiled_eval_matches_eager` — eager eval and compiled eval on the same inputs produce identical outputs

**Preliminary implementation strategy:**

New class `TestIntegrationCompiledEval` in `test_end_to_end.py`, adjacent to `TestIntegrationCompilable`. A module-level helper constructs a synthetic batch: batch=4, seq=128, attention_mask with rows of varied active lengths (e.g. [128, 100, 75, 128]). `torch._dynamo.reset()` between compiled and eager variants to prevent graph caching from hiding mode distinctions. The reproduction tests are expected to fail on the unpatched codebase — that failure confirms we have correctly identified the failure surface. Passing after the fix is the regression gate.

**Closure note:** All three tests in `TestIntegrationCompiledEval` reproduced the `InductorError: convert FlexibleLayout to FixedLayout first` at the FlexAttention boundary under compiled eval+no_grad and compiled eval+autocast. Issue confirmed. Fix (`.contiguous()` on q/k/v before `flex_attention`) is the next unit.

---

### Unit 23.B — Fix position ID resolution: replace cumsum with arange + active-token bias

**Responsibility:** Replace the cumsum-based position ID computation in `_resolve_current_position_ids` with a fresh-allocation `arange`-based approach, and install `ShramCache.total_active_tokens` to supply the per-batch active-token offset needed for correct cached-inference positions.

**Context of Correctness:**

An `InductorError: convert FlexibleLayout to FixedLayout first` was observed under compiled eval+no_grad and fp16 autocast contexts. The source was identified as `_resolve_current_position_ids`: position IDs were computed via `cumsum(dim=-1)` over the full attention mask and sliced with `[:, -current_length:]`, producing a non-contiguous stride-based view that Inductor encountered when tracing the `_make_block_mask` closure. `cumsum`-based position computation with a trailing slice cannot be used in a compiled model.

The SHRAM masking contract guarantees that within any single sequence, active tokens are left-justified — all active positions precede all padding positions. Cross-batch variation in sequence length is permitted, so different batch items may have different active lengths. Under this contract, the count of active tokens seen by a batch item across all prior forward passes equals its next available position index. This leaves room for a solution: any mechanism that produces a fresh contiguous allocation and biases a per-batch arange by the prior active-token count satisfies the positional contract for both training and inference.

Accumulating a per-batch active token count on the cache and reading it as the position bias each step is simpler and faster than alternatives such as one-hot sums or scatter-based position reconstruction. The uncached training path uses a zero bias directly. The count tensor must be pre-allocated at cache construction time and updated in-place each step; CUDAGraph captures a fixed memory graph after the first traced step and new tensor allocations in subsequent steps fall outside it.

**Invariants:**

1. `_resolve_current_position_ids` does not use `cumsum`. All tensors it produces are fresh contiguous allocations, not views.
2. For uncached calls, token at index i in batch item b receives position i if active, 0 if inactive.
3. For cached calls, token at index i in batch item b receives position (prior_active_count_b + i) if active, 0 if inactive, where prior_active_count_b is the total count of active tokens seen by batch item b across all prior forward passes through this cache.
4. `ShramCache.total_active_tokens(active_mask)` returns the per-batch active token count accumulated before this call, then updates the internal count to accurately reflect the active tokens in `active_mask`.
5. All counter tensors on cache objects are updated in-place. Creating new tensors for counter state is not permitted; all counter mutations operate on pre-allocated buffers via in-place operations.
6. `_active_token_counts` is pre-allocated in `ShramCache.__init__` as zeros with shape `(B,)`, on the same device as other cache tensors, where B is the batch size passed at construction.
7. `ShramCache.reset()` zeroes `_active_token_counts` in-place and delegates to `super().reset()` to reset all layer caches.
8. `TestIntegrationCompiledEval` passes in full: compiled eval, compiled eval+autocast, and compiled eval matches eager.

**Tests:** The three tests in `TestIntegrationCompiledEval` are the end-to-end regression gate. Unit tests for `total_active_tokens` must cover: returns zero on first call, returns correct pre-update count on subsequent calls, accumulates independently per batch item, and resets to zero after `reset()`.

**Preliminary implementation strategy:** Add `_active_token_counts` buffer and `total_active_tokens` method to `ShramCache`. Override `reset()` to zero the buffer in-place before calling `super().reset()`. Rewrite `_resolve_current_position_ids` in `huggingface.py` to accept the optional cache; when cache is present call `total_active_tokens` for the bias, otherwise use zero; build positions from `arange` offset by the per-batch bias. Pass the cache through at the call site in `forward`.

---

### Unit 23.C — Documentation: compile-mode constraints and minor documentation gaps

**Responsibility:** Document that `torch.compile(dynamic=True)` is architecturally excluded from SHRAM, and close any minor documentation gaps identified during Unit 23.B implementation.

**Context of Correctness:**

SHRAM uses token-choice routing whose output — the expert selection indices — is data, not shape. Under `dynamic=True`, torch.compile treats tensor dimensions symbolically and propagates those symbols through traced operations. Routing indices keyed on `selected_heads` values would become symbolic, making every scatter and gather operation that uses them shape-dependent. This is not a limitation waiting to be lifted; the architecture is structurally incompatible with dynamic-shape tracing. A user or maintainer who attempts `torch.compile(dynamic=True)` will encounter confusing trace failures with no indication of the underlying reason unless the constraint is stated explicitly.

The compile test suite had historically used `dynamic=True` for the `generate()` path on the assumption that varying decode-step lengths require it. This was incorrect: the cached single-token decode path is always length-1 and `dynamic=False` is appropriate and correct there too. That error is corrected as a background bug fix; this unit records the constraint in documentation so the error cannot recur.

**Invariants:**

1. `documentation.md` states that `dynamic=True` is architecturally excluded, gives the reason (routing indices are data, not shape), and distinguishes this from the supported `torch.compile(dynamic=False)` and `fullgraph=True` paths.
2. `model/README.md` includes a compile compatibility note visible to Hub users: `torch.compile` is supported with `dynamic=False`; `dynamic=True` is not supported and will produce trace errors.
3. Both documentation surfaces are accurate and do not contradict each other.
4. Any minor documentation gaps identified during Unit 23.B that fall below the threshold of their own plan entry are also closed here.

**Tests:** Documentation content is the artifact. No code tests.

**Preliminary implementation strategy:** Add a "Compile compatibility" subsection to the Important Notes area of `documentation.md` and a corresponding note to `model/README.md`. Keep both brief — the constraint statement and its one-sentence reason are sufficient.

---

### Unit 23.D — Router diagnostics: refactor return signature + add load-balance health scalars

**Responsibility:** Refactor `MoSRAHRouter.forward` to return routing decisions and routing diagnostics as structurally separated outputs, and add load-balance diagnostic scalars that characterize the load-balance mechanism rather than its outcome.

**Context of Correctness:**

Load balancing is failing and the cause is unknown. `max_vio` measures post-selection frequency imbalance — an outcome. Outcome metrics cannot distinguish cause: the same imbalanced routing frequencies are equally consistent with a bias signal that is absent, correctly sized, or overcorrecting. The load-balance mechanism operates by adding `expert_bias` to raw routing logits `xW_r`; failure has two distinct mechanical signatures — insufficient signal (bias too small to redirect routing) and excessive or misdirected signal (bias dominating or reinforcing preferences rather than correcting them). These require opposite interventions and are not distinguishable from frequencies.

The router output currently conflates routing decisions (`selected_heads`, `routing_probs`) with routing feedback (`load_balance_loss`, `max_vio`). Decisions drive downstream computation; feedback informs training and monitoring. These are structurally distinct roles. A return signature that does not reflect this distinction cannot be extended with additional feedback values without losing caller interpretability.

**Invariants:**

1. The router exposes diagnostic feedback sufficient to determine whether the load-balance signal is too weak or too aggressive relative to the routing logit scale, without consulting post-selection routing frequencies.
2. The diagnostic feedback enables determining whether the bias signal is opposing the routing preference direction (healthy correction) or reinforcing it (runaway feedback).
3. Routing decisions and routing diagnostics are structurally separated in the router's return — a caller can inspect diagnostic values without decomposing the routing internals.
4. All diagnostic feedback values carry no gradients. `load_balance_loss` retains its gradient.
5. Diagnostic values are aggregated from per-layer router outputs to model-level scalars and exposed as fields on `ShramCausalLMOutput`. `load_balance_loss` and `max_vio` preserve their existing aggregation (sum and max respectively).
6. `selected_heads`, `routing_probs`, `load_balance_loss`, and `max_vio` are numerically identical to their pre-refactor values.

**Tests:**

- Router unit: `load_balance_loss` has grad, all diagnostic scalars do not; diagnostic values have correct ranges; when `expert_bias = 0`, bias magnitude scalar equals zero and combined-logit spread equals raw-logit spread; when bias is constructed to oppose logits, alignment scalar is negative; when bias is constructed to reinforce logits, alignment scalar is positive.
- Integration: model-level diagnostic scalars equal the per-layer mean across all decoder layers.
- Regression: existing router tests for `selected_heads`, `routing_probs`, `load_balance_loss`, and `max_vio` pass unchanged.

**Preliminary implementation strategy:**

*Return signature refactoring.* `MoSRAHRouter.forward` currently returns a 4-tuple that conflates decisions and feedback. The proposed structure is `(selected_heads, routing_probs, router_diagnostics)` where `router_diagnostics` is a `dict[str, torch.Tensor]`. This groups all feedback values under a single named container, making the decision/diagnostic boundary explicit at the call site and allowing new diagnostic scalars to be added without changing the positional tuple structure. Keys: `load_balance_loss`, `max_vio`, `bias_std`, `raw_logit_std`, `logit_std`, `bias_alignment`.

*Diagnostic scalars.* Four scalars are proposed to satisfy invariants 1 and 2. All four are computed immediately after `logits = self.routing_projection(x)` and before `balance_capacity` (capacity masking injects `-1e8` sentinels that corrupt std and cosine similarity):

- `bias_std`: `expert_bias.std().detach()`. Std of the `(L,)` bias vector. Near-zero means corrections have not built up (too weak); large means corrections are significant. Interpreted relative to `raw_logit_std`.
- `raw_logit_std`: mean over `(B, N)` of per-token `logits.std(dim=-1)`, detached. Natural routing preference scale — the reference baseline.
- `logit_std`: mean over `(B, N)` of per-token `(logits + expert_bias).std(dim=-1)`, detached. Combined signal spread. Lower than `raw_logit_std` indicates the bias is flattening preferences (healthy); higher indicates amplification.
- `bias_alignment`: mean cosine similarity of `expert_bias` `(L,)` against `logits` `(B, N, L)` per token, averaged over `(B, N)`, detached. Range `[-1, 1]`. Negative: bias opposes routing direction (healthy correction). Positive: runaway feedback.

*Threading and aggregation.* Update call sites through `MoSRAH` → `ShramHybridLayer` → `ShramModel` to pass `router_diagnostics` through. `ShramModel` accumulates one dict per layer and reduces: sum for `load_balance_loss`, max for `max_vio`, mean for all four new scalars.

*Output.* Add `bias_std`, `raw_logit_std`, `logit_std`, `bias_alignment` as fields on `ShramCausalLMOutput`.

---

### Unit 24.A — Load balance loss: replace DeepSeek fixed-step mechanism with log-probability auxiliary loss

**Responsibility:** Replace the DeepSeek auxiliary-loss-free load balance mechanism with a standard log-probability auxiliary loss that scales correction magnitude with violation severity, while preserving gradient isolation to `expert_bias` only.

**Context of Correctness**

Telemetry shows load balance correction could not keep up as routing imbalance escalated. The structural cause is the DeepSeek mechanism's use of fixed-magnitude steps to update `expert_bias`: step magnitude is independent of violation severity. When routing imbalance escalates faster than the fixed step can correct, the deficit grows.

GShard-style linear losses share this weakness in a different form. A load balance signal whose gradient magnitude does not scale with violation severity can be outrun by routing concentrations that diverge nonlinearly. The correctness lesson from cross-entropy training is that log-probability signals grow as the distribution deviates from target — gradient magnitude is not constant but scales with deviation. Additionally, load balance losses that carry gradients to routing projection weights contaminate task performance learning.

The DeepSeek mechanism preserved one correct property: load balance updates reach `expert_bias` only, not routing projection weights. The correct direction preserves this isolation while replacing fixed-magnitude steps with log-probability training signals that scale with severity.

**Invariants this unit must satisfy:**

1. A factory accepting `gshard`, `ce`, and `bce` is placed in `load_balance_loss.py`. When it is invoked, it returns a loss function of the given type; all such functions have the same external contract.
2. The factory is used in `router.py` to intialize a loss function that is then used. The type of loss function is chosen from config, and is `ce` by default. 
3. The assignment probabilities are always computed by a pathway that detached logit gradients, allowing load balancing feedback to only affect the load balancing biases.
4. The `gshard` formulation computes loss as `(1/L) * Σ_i f_i * p_i`, where L is `num_mosrah_heads`, `f_i` is the detached realised routing frequency for expert i, and `p_i` is the detached-logit assignment probability for expert i.
5. The `ce` formulation computes loss as `-(1/(L-1)) * Σ_i (1 - f_i) * log(p_i)`, with `f_i` and `p_i` as above.
6. The `bce` formulation computes loss as `-(1/L) * Σ_i [(1 - f_i) * log(p_i) + f_i * log(1 - p_i)]`, with `f_i` and `p_i` as above.

**Tests**

- For each of the three formulations, verify the computed loss equals the formula numerically on known `f` and `p` values.
- Verify the factory returns a callable for each of the three valid type strings and raises for an invalid type.
- Verify `routing_projection.weight.grad` is None after backward through `load_balance_loss`.
- Verify `load_balance_loss_type` roundtrips through `to_dict`/`from_dict`; verify invalid value raises `ValueError` at config construction.
- Verify existing tests for `selected_heads`, `routing_probs`, `max_vio`, and router diagnostics pass unchanged.

**Preliminary implementation strategy (research notes — non-binding):**

- `ShramConfig` gains `load_balance_loss_type: str`, default `"ce"`, validated at construction against the supported values.
- `load_balance_loss.py` contains the three loss functions and the factory that dispatches among them.
- The router constructs the loss callable once at `__init__` via the factory and calls it during `forward`.
- `p` is computed via `softmax` applied after detaching the logit tensor, so the only differentiable path into `p` is through `expert_bias`.
- `LoadBalanceLoss` custom autograd function is removed.
- `log(1-p)` must be implemented using torch.log1p for safety.
---

### Unit 24.B — Routing logit variance reduction: near-zero scalar gate

**Responsibility:** Establish competitive magnitude between routing logits and expert_bias at initialization so the load balance mechanism is operative from step one.

**Context of Correctness**

The load balance mechanism is a competition between two magnitude scales: routing logit spread and load balance correction magnitude. expert_bias can redirect selection only when its corrections are comparable in magnitude to the logit differences that determine selection ranking. HuggingFace initialization applies standard initialization to all module parameters. `nn.Parameter` objects are exempt — they are not module weights and are not touched. After construction, `routing_projection` produces logit std ~0.4 while `expert_bias` sits at zero. The scales are orders of magnitude apart from initialization. Routing concentrates by task gradient before expert_bias builds any leverage to correct it. Unit 24.A improved the loss formulation; this asymmetry is the remaining structural cause.

A scalar `nn.Parameter` initialized near zero multiplies all routing logits before any downstream use. Because it is a Parameter, HuggingFace initialization does not override it. Near-zero initialization brings routing logit magnitude to the same scale as expert_bias at step zero. Both scales are near zero; routing starts near-uniform; load balance is effective from the first batch. The shape is scalar rather than per-head: a per-head parameter receives gradient proportional to expert usage. Underused experts receive less gradient, their scale stays suppressed, and they are excluded from routing permanently — the mechanism designed to ensure all experts contribute would instead structurally kill underused ones.

The initialization magnitude governs the timescale over which routing develops and is an architectural parameter belonging in config.

**Invariants:**
1. `MoSRAHRouter` has a scalar `nn.Parameter` `routing_scale` whose value at initialization is negligibly small relative to a standard-initialized routing projection output.
2. `routing_scale` is not a module weight and is not overridden by HuggingFace `_init_weights`.
3. `routing_scale` is applied to routing logits before all downstream use — there is no concept of logits without it.
4. `ShramConfig` has `router_init_scale: float` with a sensible positive default; invalid (non-positive) values raise `ValueError` at construction.
5. `router_init_scale` survives `to_dict` / `from_dict` roundtrip.

**Tests:**
- After constructing via `AutoModelForCausalLM.from_config` (which triggers HuggingFace init), `routing_scale` is still near zero.
- `routing_scale` is scalar (shape `(1,)`).
- `router_init_scale` roundtrips serialization.
- Non-positive `router_init_scale` raises `ValueError` at construction.

**Preliminary implementation strategy (non-binding):**
- Add `router_init_scale: float = 1e-4` to `ShramConfig` with `> 0` validation.
- In `MoSRAHRouter.__init__`: `self.routing_scale = nn.Parameter(torch.randn(1) * config.router_init_scale)`.
- In `forward`: `logits = self.routing_projection(x) * self.routing_scale` as the first logit line.

---

### Unit 24.C — Two-pathway routing architecture: semantic and load-balancing channels

**Responsibility:** Restructure the router forward pass so that expert_bias governs both selection and output contribution, eliminating the mechanism by which a selected expert can be semantically absent from the output, and separating routing computation into two explicitly named gradient channels.

**Context of Correctness**

A selected expert contributes nothing to the MoSRAH output when its routing_prob is near zero. Under the current design this is structurally possible: selection is driven by biased logits while routing_probs are gathered from unbiased logits. An expert with low unbiased preference is underloaded because the model weakly prefers it; when expert_bias redirects tokens to it, the unbiased preference and the gathered routing_prob remain low, and the expert's output contribution remains suppressed regardless of selection frequency.

Both selection and routing_probs must incorporate expert_bias. The gradient conflict — task loss must not train expert_bias, load balance loss must not train routing_projection — is resolved by two numerically identical biased logit values with complementary detach points. `semantic_logits = logits + expert_bias.detach()` drives selection and routing_probs; task gradients reach routing_projection and expert_bias is isolated from task loss. `load_balancing_logits = logits.detach() + expert_bias` drives assignment_probs; load balance gradients reach expert_bias and routing_projection is isolated from load balance loss. The router's biased/unbiased split — two numerically different values for two different computations — is replaced by two numerically identical values serving two gradient paths. All routing computation uses biased values; no unbiased routing computation exists.

`balance_capacity` injects -1e8 into over-capacity expert positions as a hard allocation constraint, not a preference signal. An over-capacity expert has high routing preference; after masking its softmax probability is near zero. Assignment_probs derived from post-capacity logits produce an inverted load balance signal, increasing expert_bias for an already-overloaded expert. Assignment_probs must be computed from `load_balancing_logits` before capacity masking.

**Invariants:**
1. `semantic_logits` incorporates expert_bias with expert_bias detached, and is the sole source for both selection and routing_probs. Expert_bias carries no gradient from task loss through this path.
2. `load_balancing_logits` incorporates expert_bias with logits detached, and is the sole source for assignment_probs. Routing_projection carries no gradient from load_balance_loss through this path.
3. `semantic_logits` and `load_balancing_logits` are numerically identical at every forward pass.
4. `selected_heads` is determined by TopK over capacity-balanced `semantic_logits`.
5. `routing_probs` is gathered from `softmax(semantic_logits)` at `selected_heads` positions and renormalized to sum to 1 per token.
6. `assignment_probs` is computed from `softmax(load_balancing_logits)` before `balance_capacity` is applied.
7. No softmax over unbiased logits exists in the router forward pass.
8. After backward on task loss only: `routing_projection.weight.grad` is not None; `expert_bias.grad` is None.
9. After backward on `load_balance_loss` only: `expert_bias.grad` is not None; `routing_projection.weight.grad` is None.
10. Module docstring and class docstring describe the two-pathway gradient architecture; all references to unbiased routing scores are removed.

**Tests:**
- Gradient isolation (task): backward on task loss only; `routing_projection.weight.grad` not None, `expert_bias.grad` is None.
- Gradient isolation (load balance): backward on `load_balance_loss` only; `expert_bias.grad` not None, `routing_projection.weight.grad` is None.
- Bias incorporated: with non-zero expert_bias, `routing_probs` differs from the zero-bias case.
- Normalization: `routing_probs` sums to 1 per token.
- Assignment_probs pre-capacity: in an inference scenario with over-capacity experts, `assignment_probs` for those experts is non-negligible — confirming no -1e8 contamination from capacity masking.

**Preliminary implementation strategy (non-binding):**
- Rename `biased_logits` → `load_balancing_logits`; introduce `semantic_logits = logits + self.expert_bias.detach()`.
- Replace `routing_scores = F.softmax(logits)` with `routing_scores = F.softmax(semantic_logits)`.
- Feed `semantic_logits` to `balance_capacity`.
- Compute `assignment_probs` from `F.softmax(load_balancing_logits)` before the `balance_capacity` call.
- `logit_std` diagnostic is numerically `semantic_logits.std()` — derivable without change.
- Update module docstring and class docstring.

---


## Unit 25 — Load balancing rebuild

The load balancing system is insufficient for
the desired task. Simply put, a global bias per element is not enough to constrain the behavior within an individual batch. There are several problems involving the way loss is taken and computed. 

### Unit 25.A: Load balancing loss fix

**Responsibility**: Fix a load balancing oversight

**Context of Correctness**

A standard mixture of expert accepts load balancing that is averaged across all batches. This makes no significant difference normally as batches can be removed by repacking before execution in expert-packed format.

This however is a mixture of attention, and cannot neglect batches. Unfortunately, this was not caught when the routing algorithm was originally written.

Based on the given telemetry, once some confusing details were understood, it is clear that load balancing is operating perfectly but imbalances between batches are causing convergence to fail. One batch which overallocates experts can be compensated for by another batch underallocating. 

This behavior must be removed, and preferably the telemetry issue fixed. Additionally, minor artifacts on p-mean and max_vio will need to be cleaned up

**Invariants**

- `router.py` is modified such that all contents to the loss function move in 1(B, N, L)` format
- `load_balancing_loss.py` is modified to accept this format
- `load_balancing_loss.py` is modified to use reductions which preserve batch dimensions where possible
- `load_balancing_loss.py` is considered in terms of using the torch builtin functions
- `router.py` and `configuration.py` and documentation has references to `p-mean` removed, with aveage used instead. 
- `max_vio.py` is computed independently in `router.py` as an independent static or class method. This method accepts the agreed [B, N, L] tensor flow and computes from there independently from the loss system.
- `forward` in router.py has primarily tensors of shape [B, N, L] or [B, N, K] flowing through the main section. Logic which does not behave this way, besides metrics and losses being orchestrated, is processed elsewhere.
- `routing_scale` is removed, and it's config entry purged, along with any needed documentation changes. 

**Tests**

- All tests are updated to be compatible
- The necessary telemetry metric function is tested as needed by virtue of being a class or static method
- 

**Preliminary implementation strategy**

- This needs a discussion to verify the plan is viable. 
- We need to stop reducing in the router itself.
- Per token frequencies of shape [B, N, L] should likely be computed in the main body then passed around to loss and max_vio functions. 
- It is almost certain metrics must be computed in helper methods to hit the forcing constraints. 
- The forcing constraints are to make refactored code saner. If you cannot meet one ,bring it up!

### 25.B: Balancing offset mechanism rebuild

**Responsibility** Rebuild the load balancing system to use a decoupled projective bias for balancing, such that the model can respond to history to maintain balance within a given batch

**Context of Correctness**

The existing load balancing system attempts to train a global offset for each expert and use that to achieve load balancing. It has become clear this is enviable

While global balancing is possible on average, individual batches tend to desire to overspecialize on a single expert and have no mechanism to correct their behavior when overusing one. This is because there is no way to modify behavior based on the history of the access patterns of the model.

The standard fix for this is to make the load balance loss modify the model alongside the main loss. however, this has been shown to have significant performance consequences. 

Instead, we will rebuild the load balancing system to use a detached projective schema that can read the activity the model has been up to in order to pull back the rate of usage. The detached training system will be retained as well for the existing benefits

**Invariants**

- The underlying routing balancing system will cease to use any bias of shape [L] as a global offset term
- The routing system will now compute logits using, in theory, the formula `logits=A@x +B@x` where A is the semantic routing matrix and B is the load balancing matrix and x the inputs for logit proejctions.
- Matrixes `A` and `B` have shape embedding, L.
- Semantic logits occurs through the construction `semantic_logits = A@x + B.detach() @ x`
- Load balancing logits occur through `load_balancing_logits = (A @ x).detach() + B@x.detach()`
- These names are properly adapted into compliant code variable names, not inserted verbatum. 

**Tests**

- The existance of matrices A and B are checked. 
- Tests are modified in any way needed to test the new system.
- Tests confirm gradients are not applied in the wrong modes to the wrong places. 

**Preliminary Implementation**

- It would be a really good idea to do some refactoring to pull out logit computations in preparation for the next unit

### Unit 25.C: Integral Routing

**Note**: 

- We will try 25.B while setting this case up.

**Responsibility**: Ensure history cannot be discarded and is taken into account when making routing decisions

**Context of Correctness**

Standard routing is inherently parallel in nature. This leaves it unable to adjust, except indirectly by observing decisions in token streams from the prior layer, to forming routing inbalances. It also is unable to tell when a set of experts has been overused and a different strategy is called for, except indirectly as discussed.

A cumulative sum of the routing logits may change this. Such a quantity, correctly constructed and used, is capable of informing on the overall preferences that have been exhibited up to this point. This may allow more advanced routing decisions for semantic purposes, and will allow more advanced routing decisions for recurrent purposes

It may, however, not be compatible with torch compile. We shall see.

**Invariants**

- The routing construct is modified to have an "integral" and "standard" control mode.
- The config has an entry added to enter integral routing mode, and is set to it by default.
- Integral mode produces two additional parameters `A'` and `B'`.
- If `logits=A@x +B@x` then the cumsum `u` is produced by shifted_logits= concat[zeros[1], logits][:-1] then `u=cumsum(shifted_logits)`
- The final logits are then produced by `logits = logits + A'@u + B'@u`
- As before, the semantic and load balance pathways are detached in separate ways
- `semantic_logits = semantic_logits + A'@u_semantic + B'.detach() @u_semantic`
- `load_balancing_logits = load_balancing_logits + A.detach() @ u_load + B' @ u_load`

**Tests**

- Tests confirm gradients work in both modes
- A test is added with a global flag at the top of the file. When engaged, it will profile the speed of both modes, in compiled or uncompiled form.
- It is checked that compilation still works in integral form.

**Preliminary strategy**

- Really think through how to design the multimode support. A stupid simple implementation will be unsupportably messy.
- It may be useful to make a "exclusive_cumsum" static method for the shifting behavior.



## 26: Load balancing fix .  
  
**Responsibility**:  
  
Fix the damn load balancing. 
  
**Context of Correctness**  
  
Split loss mechanisms have failed to correctly constrain the model. However, the SHRAM model has an extremely strong bias towards reusing the same experts again and again. Specifically, since attention can only occur within tokens in the same expert bucket, it is strongly incentivized to put all tokens in the same bucket.  
  
This presents a conundrum. We must use strong load balancing losses to constrain the model. However, if we do so, we destroy the ability to actually train towards the thing we care about because it is mostly training to satisfy the loss, not the problem.   
  
The solution is to invent a new form of loss. This is the temporal overcapacity loss.  
  
**Mathematics**  
  
All token/expert assignments are considered independently for each sequence in the batch. Let:  
  
- $z_{b, n,l}$ be the routing logit for expert `$l$` at token position $n$ in batch $b$  
- $a_{b, n,l}\in\{0,1\}$ indicate whether expert $l$ is selected by TopK at position $n$ in batch $b$  
- $d_{b, n} \in\{0,1\}$ indicates whether a token in a batch $b$ was active or not (active=1). 
- $c_{b, n,l}=\sum_{t<n} d_{b, t} a_{b,t,l}.$ is the number of experts selected cumulatively and exclusively up to token $n$ for expert $l$ in batch $b$. This can be vectorized using a cumsum.
- $S_{b, n} = \sum_{t<n} d_{b, t}$, the count of unmasked (active) tokens up to a given position.  
- $K$ be the number of experts selected per token.  
- $L$ be the total number of experts.  
- $C$ be the permitted excess above ideal allocation.  

*Imbalance Mask*

Under uniform routing, the expected number of prior assignments to each expert is $\frac{nK}{L}.$ Accommodating a total of $C$ extra expert selection allows us to claim that in a well-balanced system it should be the case that

$$
c_{b, n, l} <= \frac{nK}{L} + C
$$

Correspondingly, we can build the imbalance mask $u_{b, n, l}$ as 

$$
u_{b, n,l} = \mathbf{1}[c_{b, n, l} > \frac{S_{b, n}K}{L} + C]
$$

This selects anything that exceeds an imbalance criteria of being more than $C$ tokens ahead of the ideal load balance for possible correction. 

*Violating and Safe Sets*

An expert is 'violating' it's constraint if it exceeds it's allowed capacity and it is included in the top-k selection. We define the topk selection mask according to

$$
a_{b,n, l}  = topkmask(z_{b, n,l}, dim=l)
$$

The violating mask is given as 
 
 $$
 v_{b, n,l} = a_{b, n, l} *u_{b, n,l} 
 $$

while the compliant mask is given as

$$
\tilde{v}_{b, n,l} = !u_{b, n,l}
$$

Note $v_{b, n,l} \neq !\tilde{v}_{b, n,l}$ One additional mask of importance is the activity mask, which counts violations. This is simply given as

$$
w_{b, n} = \mathbf{1}[\sum_{l} v_{b, n, l} > 0]
$$

The activity mask is nonzero precisely when a violation occurs

*Temporal Overcapacity Loss*

The temporal overcapacity loss is defined in two steps. First is the direct statement of the loss moment itself, and second is the conditions in which it activates and the way it reduces. 

The underlying mechanics of the loss request the mean of the violating logits to reduce and the mean of the nonviolating logits to increase. Note in practice numeric epsilons are needed in the denominators to prevent division by zero. This will not affect the final result, as such situations are zeroed out later anyhow.

$$
A_{b, n} = \frac{1}{\sum_l v_{b, n, l}}[\sum_l v_{b, n,l}z_{b, n,l}] - \frac{1}{\sum_l \tilde{v}_{b, n,l}}[\sum_{l} \tilde{v}_{b, n, l}z_{b, n, l}]
$$

This has very the very useful gradient mechanics that logit redistribution is symmetric. The same amount of logit mass that is moved from one is added to another, and only violating or compliant sets see movement. The loss itself is also easy to interpret. These gradient properties can be represented as:

$$
\frac{\partial A_{b,n}}{\partial z_{b,n,l}} = \frac{v_{b,n,l}}{V} - \frac{\tilde{v}_{b,n,l}}{\tilde{V}}
$$
$$
\sum_l \frac{\partial A_{b,n}}{\partial z_{b,n,l}} = 0
$$

Left alone, however, this loss would move logit mass at all times. In reality, we simply wish to move logit mass when violations happen. For this reason we only activate the loss when a violation occurs. This is the secret behind it's effectiveness, as the loss can be strong, but will stay out of the way when no violations are occurring. The loss itself is then a reduction over active sequences, and a reduction over batches.

$$
B_{b} = \frac{1}{\sum_{n} d_{b, n}}[\sum_n d_{b,n} w_{b,n} A_{b, n}]
$$
$$
L_{TO} = \frac{1}{B}[\sum_{b} B_{b}]
$$
  
The maximum expert overclaim`$C$` allows short-lived semantic specialization. Sustained concentration activates the loss, while the loss automatically shuts off once the growing ideal allocation trajectory catches up with the expert’s cumulative usage.

### Unit 26.A — Restore Coupled Routing

**Responsibility**  
  
Restore the MoSRAH router to a single coupled routing pathway. Semantic training and load-balancing training must act on the same routing projection.  
  
**Context of Correctness**  
  
The previous routing design attempted to protect task training by sending semantic gradients and load-balancing gradients through different routing pathways. The later integral-routing extension added learned cumulative-history routing corrections. Neither have worked. It is now believed this is due to a foundational difference with respect to conventional load balancing.   
  
Unlike conventional MoE routing, SHRAM gives the task objective a direct incentive to concentrate tokens into a small number of expert buckets. Sparse attention occurs only among tokens routed to the same expert, so concentration improves short-context communication even though it damages the near-uniform bucket structure required for the architecture to scale correctly. Since this incentive is so strong, no indirect and separate gradient pathway will ever work to balance the model. No matter how hard, it is always worth it for the model to learn and evolve to trick it's routing balancer. 
As a result, it is necessary to strongly constrain the model with a loss, and to train the model to be balanced at the same time as training. This means the existing technology splitting the gradient pathways is largely superfluous. additionally, this also means the integral formulation is useless as well, and telemetry needs significant revisions due to the pending lack of bias structures. 
  
**Invariants**  
  
* The router constructs one routing-logit tensor which is sent through a loss function, then capacity balanced, then returned. 
* There is no existence of multiple detached gradient pathways. There is no concept of separate logits for different modes.
* Load-balancing gradients are allowed to flow through the router input.  
* Capacity balancing remains executed by the balancing unit, after the point the load balancing loss is applied. 
* Expert selection is determined from the capacity-balanced coupled logits.  
* The public telemetry is reduced down to simply  transitional router diagnostic contract is exactly `load_balance_loss`, `max_vio`, and `logit_std`.
* 
**Tests**  
  
  Note some level of judgment is needed. Tests which are already done should NOT be reimplemented blindly, and existing tests may be adapted to perform these roles. 
  
  - Ensure existing or new tests verify that operation with a load balancing loss over a number of batches using a synthetic training technique will eventually balance the router.   
* Verify by existing test or test addition that load-balancing gradients can reach the router input tensor.  
* Verify by existing test or test addition that routing probabilities are gathered at the selected expert indices and renormalized correctly.  
* Verify by existing test or test addition that router, model-level, and public output diagnostics expose exactly the transitional diagnostic key set.  
* Rewire significant telemetry testing throughout the test suite well outside the main test file to handle the new telemetry contract.
* Verify by existing test or test addition that eager and compiled forward/backward execution remain valid.  
  
**Audit**  
  
* Confirm by inspection that the router has one hidden-state-to-expert routing projection participating in forward routing.  
* Confirm by inspection that no independent learned balancing projection participates in forward routing.  
* Confirm by inspection that no learned cumulative-history or integral-routing projection participates in forward routing.  
* Confirm by inspection that no configuration field selects between standard and integral routing behavior.  
* Confirm by inspection that obsolete detach logic from the split semantic/load-balancing pathways has been removed.  
* Confirm by inspection that obsolete split-path diagnostics are not returned by the router, aggregated by the model, exposed in public output classes, or described as current behavior in documentation.  
* Confirm by inspection that temporal overcapacity logic has not been added in this unit.  
  
**Preliminary Implementation Strategy**  
  
* Inspect the active branch before editing. Expected current artifacts include `routing_weight`, `balance_weight`, `routing_integral_weight`, `balance_integral_weight`, `routing_mode`, `exclusive_cumsum`, `_compute_routing_logits`, and the split-path diagnostic helper.  
* - A haiku agent or brief explore over the codebase will be needed to find the telemetry checks that are being run in other units. It would be wise to treat 'what externel tests will break' as it's own pass.
* Retain `routing_weight` as the single coupled routing projection unless branch inspection reveals a specific reason to rename it. 
* Remove `balance_weight` and all `B @ x` / detached balancing-path arithmetic.  
* Update downstream aggregation and public output structures so no removed diagnostic field is still surfaced by `ShramModel`, `ShramCausalLMOutput`, tests, or documentation.  
* Remove or rewrite tests whose only purpose was to certify semantic/balancing gradient separation or integral routing.  
  
### Unit 26.B — Temporal Overcapacity Loss  
  
**Responsibility**  
  
Install temporal overcapacity loss in `load_balance_loss.py` and extend the load-balance loss factory to support loss-specific construction parameters.  
  
**Context of Correctness**  
  
Existing load-balancing losses evaluate routing primarily through aggregate expert use. That feedback is poorly matched to SHRAM, where concentrating tokens into the same expert buckets directly improves short-context communication and therefore creates a strong task-level incentive toward sustained imbalance.  
  
Temporal overcapacity loss was selected because it can identify the sequence regions where that sustained concentration exceeds an allowed trajectory and apply correction specifically there. Its centered form redirects routing preference without creating a global downward bias in the logits.  

However, significant changes are needed in the load balancing testing suite in order to accomodate the new loss. A factory system that allows usage of constructors is needed. 
  
**Invariants**  
  
* - The **Mathematics** section at the beginning of unit 26  is the authoritative definition of temporal overcapacity loss.  
* `make_load_balance_loss(loss_type, **loss_parameters)` returns a runtime callable with the established interface:  
  `loss_fn(logits, assignment_mask, active_mask) -> scalar`  
* Every registered load-balance loss is constructed through a loss-specific factory. Factories receiving unnecessary terms no-op

* Math is mapped to variables by sane names. Using the term in the math section as the name is strictly forbidden. Commenting as though the user will have the math section in front of them when reading is strictly forbidden.
* The implementation is broken up acceptably into helper functions and separate responsibilities. 
* The	`temporal_overcapacity_loss_factory` can expect the parameters `num_selected_heads`, `num_total_heads`, and `maximum_expert_overclaim`. 
* Existing load-balance loss types retain their established numerical behavior.  
  
**Tests**  
  
* Add tests by test addition that the temporal loss matches the Mathematics section for a series of hand-calculated example. Coverage should include:
  * no violations;  
  * one violating expert;  
  * multiple violating experts;  
  * inactive token positions.   
* Verify eager and compiled forward and backward behavior.  
  
**Audit**  
  
* Confirm by inspection that `_LOSS_REGISTRY` dispatches through factories rather than directly through runtime loss functions.  
* Confirm by inspection that temporal overcapacity loss does not construct routing assignments or reproduce router responsibilities.  
* Confirm by inspection that no configuration, router, telemetry, or public-output code is modified by this unit.  
* Confirm by inspection that the implementation remains tensorized over batch, sequence, and expert dimensions.  
  
**Preliminary Implementation Strategy**  
  
* * Modify `load_balance_loss.py` and its corresponding test module.  
* Convert `_LOSS_REGISTRY` from runtime loss functions to loss factories.  
* Extend `make_load_balance_loss` to accept shared construction-time keyword arguments and forward them to the selected factory.  
* Add factory wrappers for `gshard`, `ce`, and `bce`. These should return the existing runtime functions and ignore construction parameters they do not use.  
* Add `temporal_overcapacity_loss_factory`, capturing `num_selected_heads`, `num_total_heads`, and `maximum_` in the returned callable.  
* Structure the temporal loss implementation around the responsibilities established in the Mathematics section. Develop a sane helper function breakdown. Do NOT code until all parties agree the breakdown is sane. 
* Commenting needs to be aggressive. Fully understand the block commenting paradigm before coding.  
  
### Unit 26.C — Temporal Overcapacity Integration  
  
**Responsibility**  
  
Configure temporal overcapacity loss as the default load-balancing mechanism, connect the router to the extended loss factory, and install telemetry measuring how often hard capacity repair changes the router's requested assignments.  
  
**Context of Correctness**  
  
Unit 26.B makes temporal overcapacity loss available, but does not connect it to the model. The router must now provide the loss with the unconstrained routing behavior it is intended to correct: the coupled logits and their pre-capacity TopK assignments.  

This loss is intended to be strong, but shut off when needed. It will have to be the model default for this model to ever have a chance of being load balancing. 
  
Hard capacity balancing remains a downstream recovery mechanism. It prevents an unusable allocation from stopping training, but may substantially rewrite the router's requested assignments. Because that intervention is otherwise invisible, the system also needs a general diagnostic measuring how much routing is being repaired rather than learned.  
  
**Invariants**  
  
* `ShramConfig` exposes `maximum_expert_overclaim` as a nonnegative integer and preserves it through serialization.  It explains it is how many tokens over ideal load balancing an expert can claim before it starts taking loss. 
* `load_balance_loss_type` defaults to `"temporal_overcapacity"`.  
* `MoSRAHRouter` constructs the selected load-balance loss by passing `num_selected_heads`, `num_mosrah_heads`, and `maximum_expert_overclaim` to `make_load_balance_loss`.  maximum_expert_overclaim is 20 by default
* The loss continues to be computed after the logit projection but before the hard load balancing. The hard load balancing allows recovery of training from extreme upsets and limits crashes, but should not be thought of as a full solution.
* Existing loss propagation, loss weighting, routing probabilities, capacity enforcement, and final assignment behavior remain unchanged.  

**Tests**  
  Existing test suite is largely sufficient, as this is an integration suite. The only new tests needed are
  - Verification in extremely tight packing situations intervention rate is nonzero on some rounds
  - Verification that a short synthetic model will stabilize routing when used with the new loss. 
  
**Audit**  
  
* Confirm by inspection that the router does not duplicate temporal overcapacity mathematics already owned by `load_balance_loss.py`.  
* Confirm by inspection that the hard capacity-repair path does not feed its internal masking back into the load-balance loss.  
* Confirm by inspection that no temporal-overcapacity-specific telemetry is added.  
* Confirm by inspection that no loss registry or factory implementation is rebuilt in this unit.  
  
**Preliminary Implementation Strategy**  
  
* In `MoSRAHRouter.__init__`, the loss function will need to have the additional terms added:
  * `num_selected_heads=config.num_selected_heads`;  
  * `maximum_expert_overclaim=config.maximum_expert_overclaim`;  
     *`num_total_heads = config.num_mosrah_heads` 
    into the factory call installed by Unit 26.B.  
* Minor telemetry updates may be needed. 
- Other changes will likely be needed.

## Unit 27 — Causal Imbalance Loss  
  
6/10/2026  
  
**Responsibility**  
  
Implement the Causal Imbalance Loss  
  
**Context of Correctness**  
  
The last version of the load balancing loss had problems.  
  
While it did indeed initially appear that turning up the gain sufficiently was enough to balance any situation, further investigation revealed that in fact given enough training, the temporal imbalance loss would ultimately collapse at any strength. Something was wrong with the foundations of the loss itself.  
  
It turns out it was the assumption the loss could be corrected by greedy reallocation.  
  
While the temporal imbalance loss was predicated on the idea that maintaining balance was a matter of redirecting towards better options at the timestep that the imbalance occurred, this was underconsidered. It is possible for scenarios to occur in which no valid alternative move exists within the timestep, and instead the chain of action leading up to the situation has to change. Because of this flaw all training runs, eventually, settled into an equilibrium where the lack of valid moves inhibited correction.  
  
For this reason, a mere greedy temporal loss is insufficient. A causal overcapacity loss is needed instead, encouraging the model to take another path than the one that painted it into a corner.   
  
  **Mathematics**

The underlying idea behind the mathematics used for this loss are to treat the sequence of actions leading up to a violation as a joint probability in logspace. We then ask to reduce the probability of the violating chain of events. Care is taken to keep the loss interpretable in terms of nats.

The loss is designed to be hard. It pushes strongly when violating. But it also is designed to turn off. Once violations seize it will just stay out of the way not influencing the gradients at all.

*Setup*

Let Batch size be $B$, sequence length be $N$, head (expert) count be $L$, with selected experts $K$ The loss consumes:

-   routing logits $z \in \mathbb{R}^{B \times N \times L}$ — the only gradient-carrying input,
-   selection indicators $A \in {0,1}^{B \times N \times L}$, $A_{n,\ell} = 1$ iff token $n$ selected head $\ell$,
-   active-token mask $m \in {0,1}^{B \times N}$,
-   ideal per-head rate $M = K/L$ and integer slack $C \ge 0$.

We omit the batch dimension $b$ from display for brevity. Two important additional quantities are defined here. We define the log probability across timestep n as.

$$ \log p_{n,\ell} = \log \operatorname{softmax}_\ell(z_{n,\cdot}) $$

These are the log probability being bid on expert (n, l). Note this does not strictly corrolate to a standard action system as by virtue of K multiple actions are taken at once. 

We additionally require an accurate count of active tokens which were selected, tokens which could have been selected for normalization purposes, and a mask showing tokens which were both active and selected

$$ \tilde{A}_{n,l} = m_n A_{n, l} \qquad S_{n,\ell} = \sum_{i \leq n}  \tilde{A}_{i,l}   \qquad T_n = \sum_{i \leq n} m_i. $$

Note these sums should be handled carefully numerically. 

*Mask Construction**

A set of three very important masks are needed in order to correctly construct this loss. Those masks are the violations mask  $V_{b, n,\ell}$, the violations at positions mask $g_{b, n}$, and the violations in sequence mask $e_{b}$. They respectively indicate at what (b, n, l) we exceeded the allowed expert budget , whether at the indicated batch and sequence any violations occurred, and whether violations in the sequence occurred anywhere at all. 

Conceptually, the violations mask maintains that the rate of expert usage should be proportional to $\frac{K}{L}$ when well balanced. As such ,it allows at most C extra experts from the ideal ratio $M$ before considering it a violation. 

The budget at $n$ and the violation mask are

$$ \tau_n = M*T_n + C$$
$$\qquad V_{n,\ell} =\mathbb{1}\left[S_{n,\ell} > \tau_n\right] \wedge A_{n, \ell} \wedge m_{n} $$

This considers violations locations of (n, l) that are overbudget and get selected on an active token. The batch gate $g_b$ then simply verifies if any violations occurred at all within the sequence

$$
g_{b, n} = \text{any}_{l}(V_{b, n, l}) \qquad e_{b} = \text{any}_{n}(g_{b, n})
$$

These masks will then be utilized to appropriately select and reduce the right elements of the trajectory. 

*The Trajectory*

The heart of this loss is the trajectory. The trajectory is a measure of the nats of the sequence of events involving selecting only a certain expert up to a certain point. It should be handled carefully for numeric safety. The central idea of this system is that when the trajectory leads to a violating outcome, the nats should be adjusted to make that trajectory less likely. This imposes a causal prior on the system.

Each head accumulates the log-probability it was selected with, over its own selections, inclusively. It is then divided by the number of valid selections up to the point to get the nats of the sequence

$$ W_{n,\ell} = \sum_{i \le n} \tilde A_{i,\ell}\log p_{i,\ell},  \qquad E_{n,\ell} = \frac{W_{n,\ell}}{\max(S_{n,\ell},1)}. $$


$E_{n,\ell}$ is the mean log-probability with which head $\ell$ has won its selections so far. The accumulation is inclusive as part of being selected in top k is the current step's probabilities. 

*Contrast*

We then form a contrast between the typical nats at location n, and the violating quantities. Specifically, the contrast between the mean of the violators and the mean overall at sequence position n.

$$
Q_{n} = \frac{\sum_{l} V_{n, l} E_{n, l}}{max(\sum_{l} V_{n, l}, 1)} \qquad \bar{Q}_{n} = \frac{\sum_{l} E_{n, l}}{L}
$$
The core contrast driving the entire loss is then 

$$
F_{n} = Q_{n} - \bar{Q}_{n}
$$

This measures, in essence, at this step $n$ the difference between the nats of the violating set vs the typical set, and thus encourages gradients to raise the typical outcome while lowering the outcome for the violating set. This is equivalent to looking at a probability ratio between them.

We now reintroduce the batch dimension for analysis and discussion during reduction

$$
F_{b, n} = Q_{b, n} - \bar{Q}_{b, n}
$$

*Reduction*

Reduction is by mean and only includes comparisons with active violations. Critically, to avoid dilution of gradients, reduction only ever selects elements with actual violations to correct. 

 The first reduction is into sequence form and forms a mean out of the elements with valid contrasts. If there are no violations, that comparison is not included. This prevents unfair biasing of an anchor with nothing to contrast it with.

$$
D_{b} = \frac{\sum_{n} F_{b, n}*g_{b,n}}{\text{max}(\sum_{n} g_{b, n}, 1)}
$$
The second stage of reduction again only includes cases where there is actually a violation. This finally produces the loss itself

$$
\mathcal{L}_{CO}  = \frac{\sum_{b} e_{b} * D_{b}}{\text{max}(\sum_{b} e_b, 1)} 
$$

*Properties**

1. **Exact inactivity.** When no violations occur, the loss contributes exactly zero and produces no gradient. This allows the loss weight to be large without affecting already-valid routing behavior. This loss was designed as a strong constraint.
2.  **Causal credit assignment.** A violation at position $n$ penalizes the selected trajectory that led expert $\ell$ to exceed capacity, not merely the local choice at $n$. Earlier selected uses of the same expert receive gradient through the cumulative log-probability term telling them to be less confident
3. **Log-space persistence.** While a violating trajectory remains selected, reducing its log-probability does not suffer the same vanishing behavior as directly penalizing probability. The penalized selected log-probability retains a bounded, non-vanishing gradient as $p_{n,\ell}$ becomes small. This

4. Exact inactivity. This loss is completely off with no gradient when no violations occur. This allows one of the required properties to be fulfilled; the loss can be extremely strong but not sabotage training.
5. Persistence. While active, probabilities cannot get arbitrarily small to shut off gradients as we operate in logspace. 
6. Interpretability. The loss should indicate how many nats more the violating choices tend to be. 

  
**Invariants**  
  
- Install the Causal Overcapacity Loss in `load_balance_loss.py`  
- Loss is accessed through mode `causal_overcapacity`  
- Default loss mode is `causal_overcapacity` in config.  
- Maintain code quality standards at level of prior entry.  
- Variable names are not as in the math, but have been mapped to sane names in the code.  
- Comment quality is high, and includes docstring for contracts, an algorithm overview, then blocks with good commentary explaining the why, not the how. Code is otherwise largely self-documentating due to organization and good variable naming.  
- Numeric clamping has been properly thought through.   
  
Make sure to consult the imbalance overcapacity loss to see how to do this correctly.  
  
**Tests**  
  
  * Add tests by test addition that the temporal loss matches the Mathematics section for a series of hand-calculated example. Coverage should include:  
  * no violations;    
  * one violating expert;    
  * multiple violating experts;    
  * inactive token positions.     
* Verify eager and compiled forward and backward behavior.  
  
**Preliminary Implementation Strategy**  
  
- Most of the effort of writing this will not go on the page. It will require carefully thinking through the design ahead of time to implement to standards. Implementing a working algorithm is one of the least significant parts of the problem. Commenting and verifying correctness is much harder.   
- log softmax should be computed with the torch kernel, not directly.
- Considerable effort should be spent mapping variables to sane names before coding. I would not write without a plan for variables  
- Consult frequently if help is needed.

---

## Unit 28 — Mechanical Load Balancing

**Responsibility**

Insert mechanical load balancing into the routing unit along with associated machinery

**Context of Correctness**

It has become clear no loss-based solution can load balance this router.

Any such solution appears to have extremely negative consequences on the ability of the model to train when the loss is turned up high enough to hold back degeneracy. An alternative was needed. 

A significant amount of expert was spend looking into alternative solutions to mechnanically load balance the router without ever using an auxiliary loss. After significant effort, one has been found which is guaranteed to effectively, rapidly, and precisely load balance the router under any situation. Specifically, when certain constraints are put on the values of the number of experts and the number of experts chosen it is possible to divide a sequence of tokens up into 'blocks' which have a guarantee that each expert fires within the block at most once. Using this guarentee, it is then possible to solve for a solution within a finite number of rounds of known length.

**Mathematics**
*Block Mathematics*

Mechanical load balancing is built around a simple exact-cover contract: each local block is shaped so that perfect load balance is equivalent to using every expert exactly once. This turns load balancing from a global correction problem into a local construction problem. The cost is a compatibility restriction between the number of experts and the number of experts selected per token.

Let $z$ be the routing logits, let $E$ be the number of experts, let $K$ be the number of experts selected per token, and let $W$ be the number of tokens in a mechanical routing block. Let $M = f(z)$ be the on/off expert-selection mask produced by some block solver $f$, where $M_{n,e}=1$ means token $n$ selected expert $e$.

This unit accepts the compatibility constraints

```text
E % K = 0
W = E / K
```

so each block contains exactly

```text
W K = E
```

expert-selection slots. Since there are also $E$ experts, the block is perfectly load balanced exactly when each expert appears once in the block. This immediately implies if $f$ constructs $M$ to select each expert only once, it also will be perfectly balanced. 

*Causal Construction*

In order for this to be usable with a decoder model, a causal method of construction is needed for inference compatibility. This can be achieved remarkably easy. 

1) Start from a set of token logits
2) Pad to block length
3) Reshape into blocks of length W
4) Create a boolean mask filled with false
5) For length of W across each collection of logits in the block, causaly
   6) Mask out the experts already chosen by setting their logits to -inf
   7) Choose the topk scores and indexes
   8) Set those indexes to now be used
9) Reshape back to standard form
10) Discard any padding
11) Return routing scores, and routing indices.

Note that this will require the introduction of a routing cache to keep track of the hidden state
for inference mode. Nonetheless, this is entirely achievable. We also note that while it was experimented. This does not guarentee global optimality, and instead takes the best action at each causal step.


### 28.A -- Training Mode 

**Responsibility**

Install the mechanical load balancing itself within the router

**Context of Correctness**

It is possible to produce ideal load balancing in training mode by enforcing a one-usage-per-block rule with additional installed constraints

**Invariants**

- Config is modified to assert that (E % K = 0) and explain clearly that if not true it needs to be true to achieve load balancing.
- The algorithm is implemented inside the router and operational in a training mode. 
- The algorithm is implemented in a block parallel, efficiently compilable manner.

**Tests**

- Training mode tests pass
- Inference tests may still be broken, as caches have not yet been modified to accommodate this paradigm
- Training tests should verify, by tracking max vio, that exact load balancing is occurring. 

**Preliminary Implementation Notes**

- It was found that the serial variant was the fastest variant available, and still very fast, under 1 ms.
- W, the block length, can be computed as E/K with the assertion in place.

### 28.B -- Regret Loss

**Responsibility**

Provide the model with the ability to refine the moments it chooses experts
by means of providing it with a 'regret' loss that can let it know when 
it wanted an expert later it used prematurely

**Context of Correctness**

In theory, if the model could perfectly balance it's usage of tokens, we might expect that for an ideal balancing situation the model would always allocate the most probability to the cases where it uses it's experts. Failing to do so means 

**Invariants**

- The regret loss is the only type of loss available
- The regret within a block, at a particular expert configuration, is defined as max(p_maximum - p_chosen, 0), including only tokens which are unmasked. 
- The regret loss is the mean of this, adjusted appropriately to ignore masked tokens. 
- The regret loss computation is installed as a static method in the router class
- The regret is returned by the router, and reduced by mean.

**Tests**

- Verify regret computation is correct by comparing statically computed values to actual returns
- Verify regret correctly ignores masked tokens.
- Verify router now returns a regret loss.

**Preliminary Implementation Notes**

Working tensor is `(B, num_blocks, L)` — one regret scalar per batch item, block, and expert.

Sources: `routing_scores` (raw softmax over all L, not renormalized routing_probs) and
`selected_heads_blocked` (already available from the block solver for-loop — no recomputation).
Both are padded and reshaped into block form to match the solver's layout.

Gating is at block level: a block is live iff it has at least one active token. All L experts
in a live block count toward the mean. Within a live block, the active mask gates both the
p_maximum search (only active tokens are candidates) and the p_chosen lookup (an expert
assigned to a dead token is treated as having p_chosen = 0).

The static method receives routing_scores, selected_heads_blocked, active_mask, and the block
geometry (num_blocks, W, L) — all available at the call site in forward().

### 28.C -- Inference and Caching

**Responsibility**

Install inference support and necessary caching

**Context of Correctness**

The algorithm used for load balancing is stateful, and thus requires caching. Specifially, within a block state is needed to know, for the next iteration, how many experts have already been exhausted within the block. 

**Invariants**

- A new cache has been placed in the cache folder called router_cache.py
- The cache is now installed in shram_layer_cache as an additional term.
- The cache is static, compilable, and preallocates memory for all possible terms then fills them 
  in statically as inference runs. 
- The cache is passed in as an instance in routing.
- Routing accepts a None argument to distinguish training, like all other caching subsystems. 
- The cache can be used to get the necessary information to continue generation or do a new prefill session

**Tests**

- All original tests should now pass.
- When inspected, max vio should be perfectly balanced

**Preliminary implementation strategy**

- The cache should likely be made as a bool array of inference_sequence_length or the appropriate term on the config
- It is likely easiest to just record the entire chain of states up to the current point. 
- The cache should use a tensor int to track how many tokens deep the cache has already traveled, to know where to insert new tokens
- Caches of this nature MUST me updated in-place, including counters and such, using tensors, to meet all compile requirements. 
- It might be useful to consult the other caches for motivation and to know the requirements and available patterns


### 28.D -- Cleanup Identification

**Responsibility**

Get ready to Clean out old loss systems and tests, and install additional telemetry in their place

** Context of Correctness **

Many groups of tests and router functionality concern loss, telementry, and tests that are no longer applicable now load balancing is exact. This should be identified and removed.

Because it is such a foundational part of the assumptions up to now, the issues may be significant and found all over the repo. A dedicated step to gather places that depend on that assumption is needed.

**Invariants**

- Explore steps were taken to identify the locations depending on this assumption deserving of adjustment
- The entire load balancing loss unit has been examined and is planned for removal
- The relevant telemetry has been planned for removal, as has any config entries and reduction technology that will no longer be needed. 
- Documentation, docstrings, and such have been examined for revision
- Tests that are no longer needed have been planned for removal
- The whole "overallocation" system and it's tests can now be removed and replaced with the exact balance + W. 
- The revision has been inserted concretely into 28.D -- Cleanup Execution.
**Tests**

None

**Preliminary strategy**

- Use explore agents. Attack the problem from multiple angles
- It is likely useful to read the entirety of plan.md, and to have explore agents do so too.
- Don't expect one iteration of corrections to restore a sane repo state

### 28.E -- Cleanup Execution + Telemetry

**NOTE: This plan entry is a generated artifact produced by 28.D (Cleanup Identification).
It is not held to normal plan-entry standards. Its authority comes from the 28.D research
session, not from independent reasoning. The checklist below was generated by aggressive
Haiku agents — it guarantees that the listed issues EXIST in each file, but the exact
nature of each fix requires reading the file first.**

**Folds in 28.F (Revised Telemetry) — logit_regret surfaced in ShramCausalLMOutput.**

---

**Source file changes (COMPLETE):**

`src/shram/model/attention/load_balance_loss.py`
- Delete entire file (no imports, no callers anywhere in src/)

`src/shram/model/model.py`
- Rename `total_load_balance_loss` accumulator → `total_regret_loss`
- Rename `layer_diagnostics["load_balance_loss"]` → `layer_diagnostics["regret_loss"]`
- Add `total_logit_regret` accumulator; aggregate `layer_diagnostics["logit_regret"]` (mean across layers, like logit_std)
- Remove `max_vio` accumulator and aggregation entirely
- Update output dict: `"load_balance_loss"` → `"regret_loss"`, add `"logit_regret"`, remove `"max_vio"`
- Update module docstring and forward() docstring

`src/shram/model/huggingface.py`
- ShramCausalLMOutput: rename field `load_balance_loss` → `regret_loss`; add field `logit_regret`; remove field `max_vio`; update comment
- forward(): rename `backbone_outputs["load_balance_loss"]` → `backbone_outputs["regret_loss"]`; populate `logit_regret=backbone_outputs["logit_regret"]`; remove `max_vio=` population
- Update forward() docstring

`src/shram/model/attention/mosrah.py`
- Docstring only: update stale diagnostic key list to reflect {regret_loss, logit_regret, logit_std}

`src/shram/model/configuration.py`
- Remove `mosrah_overallocation_factor` param, validation block, and docstring entry
- Remove `max_bid_rounds` param, validation block, and docstring entry
- Update `mosrah_packed_length` formula: remove `* self.mosrah_overallocation_factor` (exact balance → training_sequence_length * num_selected_heads // num_mosrah_heads)
- Update `mosrah_cache_length` formula: same removal

`src/shram/documentation.md`
- Line 214: update "load-balance loss" → "regret loss", remove "MaxVio"

---

**Test file changes (REMAINING):**

**Known post-28.E issue (user-flagged):** Removal of `mosrah_overallocation_factor` required
adding explicit block-length padding room to the packed length. User added this after src
files were completed. May break some tests — if tests fail on packed-length assertions,
this is the likely cause. Investigate before assuming the rename work broke anything.

**Protocol for each file:** Read the file first. Scan for ALL occurrences of removed
symbols: `load_balance_loss`, `max_vio`, `mosrah_overallocation_factor`, `max_bid_rounds`,
`balance_capacity`. Fix each appropriately — rename, remove kwarg, or delete surrounding
class if the concept it tests no longer exists. Do not assume the agent checklist captures
every occurrence or correctly identifies the fix.

`tests/shram/test_end_to_end.py`
- Known issues: `load_balance_loss`/`max_vio` in TestIntegrationRouterDiagnostics;
  `mosrah_overallocation_factor`/`max_bid_rounds` in TestIntegrationCapacityEnforcement
  and _save_base_model; exact fixes determined on read.

`tests/shram/model/attention/test_shram.py`
- Known issues: `load_balance_loss`/`max_vio` references; scan for removed params too.

`tests/shram/model/test_model.py`
- Known issues: `load_balance_loss`/`max_vio`; add `logit_regret` assertions; scan for removed params.

`tests/shram/model/test_decoder_layer.py`
- Known issues: `load_balance_loss`/`max_vio`; scan for removed params.

`tests/shram/model/attention/test_mosrah.py`
- Known issues: `load_balance_loss`/`max_vio`; update docstring; scan for removed params.

`tests/shram/model/test_huggingface.py`
- Known issues: `load_balance_loss`/`max_vio`; add `logit_regret` field check; scan for removed params.

`tests/shram/model/test_configuration.py`
- Known issues: delete TestMosrahPackedLength class (7 tests); delete TestMaxBidRounds class
  (5 tests); trim test_roundtrip_preserves_all_fields (remove mosrah_overallocation_factor=1.3
  kwarg + 2 assertions). Scan for additional removed-param usage.

`tests/shram/model/attention/test_expert_packing.py`
- Known issues: overflow error message match strings reference removed param name; scan
  for additional removed-param usage.

### 28.F -- Revised Telemetry

**Responsibility**

Provide some additional telementry on regret

**CoC**

When an expert is utilized, it means that expert can no longer be used later. In a block, an 'ideal' solution uses the expert where it is wanted the most. Thus, it is possible to measure the "regret" a model feels in terms of how strongly it preferred an expert when chosen vs how strongly it would have preferred the expert at the location of maximum preference.

**Invariants**

- `logit_regret` and `prob_regret` are present in `ShramCausalLMOutput` as finite detached scalars.
- `max_vio` is removed from `ShramCausalLMOutput` — exact load balancing makes it permanently zero and therefore misleading. Removal is executed in 28.D.
- Documentation is updated to describe both metrics and their interpretation.

**Tests**

- Verify `logit_regret` and `prob_regret` are present in `ShramCausalLMOutput` and are finite detached scalars.
- Hand-calculated case where every expert is assigned at its ideal position: verify both metrics match analytically derived expected values.
- Hand-calculated case where assignment is forced away from ideal: verify both metrics match analytically derived expected values.

**Preliminary Implementation Strategy**

- Compute both metrics inside the router after block assignment is resolved, reusing the logit tensor and assignment indices already available there.
- For `logit_regret`: normalize per-expert logit at the assigned position by the within-block mean and std for that expert (z-score style); mean over experts, blocks, and batch. It might be as simple as the highest logit score compared to the chosen one. 
- For `prob_regret`: apply softmax over the logit tensor; subtract assigned-position probability from ideal-position probability; mean over experts, blocks, and batch. Again, highest to chosen.
- Surface both through the same diagnostic return path as other router scalars, aggregate across layers in `ShramModel`, expose in `ShramCausalLMOutput`.
- `logit_regret` is expected to be in [0, inf].
- `prob_regret` is expected in [0, 1].


## Unit 29.A — Split residual gates in DecoderLayer (vector)

**Note:** Gradient norm explosion discovered in training. Root cause: a single `residual_gate`
parameter was shared between the attention and MLP residual connections in `DecoderLayer`.
Gradients from both sublayers accumulated into the same parameter simultaneously, producing
compounding norm growth at depth.

**Fix:** Split into `attn_residual_gate` and `mlp_residual_gate` — two independent parameters
with the same near-zero initialisation. Gradients from attention and MLP residuals are now
isolated. `test_decoder_layer.py` updated at the one test site that opened the gate directly.

**Invariants:**
- `DecoderLayer` owns two independent gate parameters: `attn_residual_gate` and `mlp_residual_gate`.
- No shared gradient path exists between the two sublayer residual connections.

## Unit 29.B — Scalar residual gates

**Note:** Vector gates (512 params each) produced large, noisy gradient norms dominated by
`mlp_residual_gate` across all layers. Root cause: each of the 512 gate elements receives a
gradient equal to the MLP output at that embedding dimension, summed over batch and sequence.
MLP outputs (SwiGLU) have larger magnitude than attention outputs, amplifying the effect.
A vector gate tries to learn per-dimension scaling on top of what the MLP projection already
controls — the directional question is already answered by the MLP weights. A scalar gate
learns only "how much should this sublayer contribute," a single clean gradient signal.

**Fix:** Both gates changed from `[embedding_width]` to scalar (shape `[1]`, init: zero).

**Invariants:**
- `attn_residual_gate` and `mlp_residual_gate` are scalar parameters (shape `[1]`, init: zero).
- Gradient into each gate is a single scalar — no per-dimension gradient splitting.

## Unit 29.C — Config-selectable residual gate vs fixed scale

**Note:** Even with scalar gates (29.B), gradient norms on `mlp_residual_gate` remained
large. The gate itself may not be needed now that mechanical load balancing removes the
router instability that motivated skip-init. Added `use_residual_gate: bool = True` to
`ShramConfig`. When `False`, each DecoderLayer uses a fixed `1/√num_decoder_layers` scale
on both residual branches — no learnable parameter, O(1) variance at depth by construction.
Default `True` preserves existing behavior.

**Invariants:**
- `ShramConfig.use_residual_gate` is a boolean param, default `True`, survives roundtrip.
- `DecoderLayer` with `use_residual_gate=True` owns `attn_residual_gate` and `mlp_residual_gate` scalar params.
- `DecoderLayer` with `use_residual_gate=False` owns no gate params; applies fixed `1/√L` scale.

---

## Unit 30 — Final Audit

This unit can only be completed once the paper is done and results are back. It involves resolving the paper to the codebase. 

**What:** Review every audit note in plan.md and, if not overridden, ensure compliance. Crosscheck with papers/main.tex and verify whether or not paper is still complaint.

**Invariants this unit must satisfy:**
- Every invariant in job.md is satisfied and has a corresponding test.
- No file has hardcoded architectural parameters.
- All documentation standards are met.
- Every decision recorded in this plan has a corresponding entry in `documentation.md`.
