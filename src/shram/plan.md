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
- [ ] Unit 5 — Local sliding-window attention module (h_l)
- [ ] Unit 6 — Expert packing and unpacking: permutation machinery, padding, masks
- [ ] Unit 7 — Bottlenecked Ensemble Attention (BEA): per-head attention on packed tensorsHJ
- [ ] Unit 8 — MoSRAH sparse path: routing → packing → BEA → unpacking → weighted reduction
- [ ] Unit 9 — SHRAM hybrid layer: assemble H(x) = h_l (Unit 5) + h_s (Unit 8)
- [ ] Unit 10 — DecoderLayer: replace attention sublayer, propagate load_balance_loss
- [ ] Unit 11 — ShramModel: aggregate load_balance_loss in output
- [ ] Unit 12 — ShramForCausalLM: expose load_balance_loss; KV cache resolution
- [ ] Unit 13 — upload_to_hub
- [ ] Unit 14 — documentation
- [ ] Unit 15 — end-to-end tests
- [ ] Unit 16 — final audit

---

## Status

**Current state:** Unit 4 complete. MoSRAHRouter and LoadBalanceLoss verified with 23
passing tests. Unit 5 (local sliding-window attention module) is next. Network tests
deselected (Hub repo not yet created).

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
  selects K of L heads and only those heads' caches are updated. Standard DynamicCache is
  used via virtual layer indexing: each SHRAM layer computes its own virtual indices as
  `layer_idx * num_mosrah_heads + head_idx`. Total virtual layers = num_hidden_layers *
  num_mosrah_heads. Each layer manages its own cache slots directly without ShramForCausalLM
  coordination. If virtual layer indexing proves unworkable, fall back to a custom cache class.
  Must implement before Unit 11.
  Resolution: RESOLVED — virtual layer indexing per SHRAM layer.

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

### Unit 5 — Local Sliding-Window Attention Module

**Invariants this unit must satisfy:**
- Implements h_l(x) as a standalone module: (B, N, d) → (B, N, d). This module is a
  verified black box before the hybrid layer (Unit 9) assembles it.
- Local attention is enforced by the kernel natively. The window size is passed as a
  parameter to the kernel itself; masking is handled internally. A boolean attn_mask
  constructed outside the kernel is not acceptable (job.md Architecture §).
- Window size is configurable via config. Nothing is hardcoded.
- The module is causal within the window.
- The specific kernel (flash_attn, flex_attention, or other) is determined at this unit's
  planning time. The unit planner evaluates available options against the kernel-native
  invariant above and records the choice and rationale here before implementation begins.
  OD-1 is closed at this point.
- Tests verify: output shape, tokens outside the window receive zero attention weight,
  causal ordering is preserved within the window, window_size equal to sequence length
  degrades to standard causal attention.

**Paper refs:** Appendix A.Local Attention. **job.md ref:** Architecture § (local path).

---

### Unit 6 — Expert Packing and Unpacking

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

### Unit 7 — Bottlenecked Ensemble Attention (BEA)

**Invariants this unit must satisfy:**
- BEA operates on `packed_hidden` (B, L, T, d); output y (B, L, T, d).
- W_Q, W_K, W_V have shape (L, d, `mosrah_head_dim`); W_O has shape (L, `mosrah_head_dim`,
  d). Each of L heads has independent parameters — no weight sharing across heads.
- RoPE is applied with the supplied position tensor (either `packed_positions` for
  main-sequence mode or local slot indices for semantic-sequence mode). BEA passes the tensor;
  it does not choose the mode — that is the caller's responsibility.
- Causal masking is a triangular mask over the T (packed) dimension.
- Padded positions (`active_mask` == False) do not contribute to attention output.
- `attention_scaling` returned by RoPE is applied to the attention logits.
- Tests verify: output shape, head independence (perturbing one head's weights changes only
  that head's output), RoPE passthrough (both position tensors produce different outputs),
  padding positions do not influence real-token outputs, causal masking holds.

**Potential blocker:** OD-4 (resolved in principle — extend rope.py for arbitrary position
tensor shape). If rope.py modification is not yet done, push it as a blocker unit when Unit 7
planning begins.

**Paper refs:** Appendix A.BEA, Appendix B.RoPE Mechanics.

---

### Unit 8 — MoSRAH Sparse Path

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

### Unit 9 — SHRAM Hybrid Layer

**Invariants this unit must satisfy:**
- SHRAM forward: H(x) = h_l(x) + h_s(x), where h_l is the local sliding-window module
  (Unit 5, verified black box) and h_s is MoSRAH (Unit 8, verified black box).
- h_l and h_s have fully independent parameters. Their outputs are summed.
- load_balance_loss from h_s is propagated through the hybrid layer's return.
- Output shape: (B, N, d) — same interface as GroupedQueryAttention it replaces.
- Tests verify: output shape, independence of paths (zeroing h_l parameters leaves only
  h_s contribution and vice versa), load_balance_loss is present in the return.

**Paper refs:** §3 Design Success Conditions.

---

### Unit 10 — DecoderLayer Update

**Invariants this unit must satisfy:**
- DecoderLayer uses SHRAM hybrid layer in place of GroupedQueryAttention.
- load_balance_loss returned by the SHRAM layer is propagated through DecoderLayer's output.
- The pre-norm + residual structure is unchanged: x' = x + SHRAM(RMSNorm(x));
  x'' = x' + FFN(RMSNorm(x')). This structure transfers unchanged per Appendix A.
- Tests verify: output shape unchanged, load_balance_loss present in output, structure holds.

---

### Unit 11 — ShramModel Update

**Invariants this unit must satisfy:**
- ShramModel collects load_balance_loss from all decoder layers and includes an aggregated
  scalar in the output dict.
- The existing output dict keys (last_hidden_state, past_key_values, hidden_states) are
  preserved with unchanged semantics.
- Tests verify: load_balance_loss present in output dict, aggregation is correct (not just
  the last layer's value — it must reflect all layers), existing output unchanged.

---

### Unit 12 — ShramForCausalLM Update

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

### Unit 13 — upload_to_hub.py

**What:** Adapt the upload script for the SHRAM model type. Update class names, model type
string, and model card content. Register `ShramConfig` and `ShramForCausalLM` with the AutoClass
API and push all model files to the Hub.

**Invariants this unit must satisfy:**
- A freshly instantiated model can be round-tripped: upload → `from_pretrained` → forward pass.
- The model card accurately describes the SHRAM architecture.
- No weights are uploaded. No checkpoint is assumed.

---

### Unit 14 — Documentation

**What:** Write `documentation.md` covering design decisions, deviations from the paper, and
limitations. Update `README.md` with accurate architectural details. Record every open decision
resolved during implementation and the rationale for each.

**Invariants this unit must satisfy:**
- Every open decision resolved during implementation is recorded with its rationale.
- Limitations are documented explicitly, including any known train/inference mismatches.
- The model card accurately describes the SHRAM variant.

---

### Unit 15 — End-to-End Tests

**What:** Full-stack smoke tests: instantiate from config, run a training step, verify loss
decreases. Include load-balance loss in the training step. Include network tests for the Hub
round-trip.

**Invariants this unit must satisfy:**
- The model can be instantiated from a config, run a forward pass, compute loss, and backpropagate
  without error.
- The load-balance loss is accessible from the forward output and participates in the backward pass.
- Network tests verify the Hub round-trip.

---

### Unit 16 — Final Audit

**What:** Review every file in `src/shram/` against the invariants in `job.md`. Verify no
hardcoded values, no missing documentation, no gaps between tests and intent. Apply the
close-the-testing-gap rule to any defect found.

**Invariants this unit must satisfy:**
- Every invariant in job.md is satisfied and has a corresponding test.
- No file has hardcoded architectural parameters.
- All documentation standards are met.
- Every decision recorded in this plan has a corresponding entry in `documentation.md`.
