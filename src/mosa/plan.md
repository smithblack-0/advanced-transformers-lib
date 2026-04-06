# Implementation Plan: advanced-transformers-lib — MoSA Baseline

## What This Document Is

This is the planning and process record for the LLM-assisted implementation of the MoSA hybrid
attention variant. It serves the same function as the Llama 3 plan: active development tool and
trust artifact. Every unit is planned here before code is written, every decision is recorded for
human review, and every unit is verified here before the next begins.

The process it records:

- Before writing any code for a unit, the plan for that unit was written and reviewed.
- Each unit was implemented, tested, and verified before the next began.
- When a blocker arose, it was pushed onto a stack, resolved as its own verified unit, and then
  the interrupted unit resumed. No blocker was hacked around inline.
- All significant decisions were surfaced for human review. The human author approved or corrected
  each one. Autonomous decisions were reported, not hidden.

---

## Completion Record

- [x] Preliminary — copy llama3 to mosa, rename, verify baseline tests pass
- [x] Unit 1 — audit the copy against MoSA requirements; produce ordered change list
- [ ] Unit 2 — configuration.py (add MoSA params; resolve open decision: param names/defaults)
- [ ] Unit 3 — attention.py (MoSA hybrid layer; resolve open decisions: GQA/MHA, routing, cache)
- [ ] Unit 4 — rope.py (add rotate_fraction support for sparse heads)
- [ ] Unit 5 — decoder_layer.py (wire up new attention class)
- [ ] Unit 6 — huggingface.py (causal mask and cache handling)
- [ ] Unit 7 — upload_to_hub
- [ ] Unit 8 — documentation
- [ ] Unit 9 — end-to-end tests
- [ ] Unit 10 — audit

---

## Status

**Current state:** Units 1 complete. Change list produced. 110 local tests pass on the mosa copy. Unit 2 not yet started.

---

## Governing Principles

This plan records not just *what* to build but *why* each decision was made, so future sessions can
make good changes without reconstructing lost context. Every design decision includes its rationale.
Every component includes the invariants its tests must enforce. If an invariant changes, the tests
must change to match before the unit can be considered verified.

The architecture is a **synthesis**, not a transcription. The MoSA paper and reference repository
condition the implementation but do not dictate its structure, style, or organisation. The Llama 3
baseline in this repository sets the code quality bar. Where sources conflict or leave choices open,
a decision is made here and flagged for user review. Where the correct answer is unknown, work stops
and the user is asked.

---

## Process Rules

These are non-negotiable. Deviating without explicit user approval is not acceptable.

**One unit at a time.** Never begin a new unit while the current one is unverified. A unit is not
complete until its tests pass and accurately describe the intended behaviour.

**Blocker stack.** When completing a unit requires work outside its scope, that work is a new unit
pushed onto a stack. Complete and verify the blocker, then return. Only make the changes needed for
compatibility — do not expand scope.

**Surface decisions.** Autonomous decisions are permitted but must be reported to the user for
review. Uncertain decisions must be escalated before proceeding. Do not resolve ambiguity silently.

**Keep this plan current.** Update the status section and unit checklist continuously. The plan must
reflect actual state at all times so work can be resumed after a session break without loss.

**Plan first within each unit.** At the start of each unit, confirm file granularity, list what
will be implemented, and identify any decision points before writing code.

**Close the testing gap.** When a defect is found — in audit or anywhere else — the resolution is
not complete until two questions are answered: (1) what invariant did the existing tests fail to
enforce, and (2) what test change closes that gap? The fix and the test correction are a single unit
of work. A defect resolved without closing the testing gap is not verified — it is just patched.

---

## Code Quality Standards

These apply without exception to every file. Clean, well-structured code is a first-class
requirement, not a means to an end. Code that cannot be trusted is wrong by definition.

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
- Skipping documentation on non-self-documenting code and writing useless line-by-line narration
  are failure modes of equal severity
- Code should be self-documenting through clear naming wherever possible; comments fill the gap
  where naming alone is insufficient

---

## Testing Philosophy

A codebase that works but cannot be verified has no value for research. Only a verified
implementation can be used to draw scientific conclusions. **Verified-but-imperfect is more
valuable than working-but-unverified** — only the former can be trusted as a research baseline.

Tests are first-class artifacts. They are written alongside the implementation, not appended
afterward. A component without passing tests is not complete regardless of how correct it appears.

**Rules:**
- Each src file has a corresponding test file mirroring the structure under `tests/mosa/`
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

### Preliminary — Copy and Rename ✓

`src/mosa/` and `tests/mosa/` copied from `src/llama3/` and `tests/llama3/`. All class names,
module references, and string identifiers renamed from `Llama3`/`llama3` to `Mosa`/`mosa`.
`model_type` updated to `"mosa"`. 110 local tests pass. `src/llama3/` and `tests/llama3/`
untouched.

---

### Unit 1 — Audit the Copy; Produce the Change List

**What:** Read the MoSA paper and reference repository against the copied mosa source. For each
file in `src/mosa/`, determine: does it transfer unchanged, does it need modification, or does it
need replacement? For each file that needs modification or replacement, describe what must change
and why, and identify any open decisions that must be resolved before that work can begin.

The output of this unit is a fully populated change list appended to this plan — one entry per
affected file, with rationale and open decisions called out. The units 2..N in the completion
record are filled in at that point. No code is written in this unit.

**Why it exists:** We do not yet know the full scope of changes required. Committing to a unit
breakdown before understanding the problem produces a plan that will be wrong. This unit produces
the evidence the plan needs before implementation begins.

**Invariants this unit must satisfy:**
- Every file in `src/mosa/` has been assessed.
- Every required change has a stated rationale grounded in the paper or reference repo.
- Every open decision is named and tied to the unit where it must be resolved.
- The completion record is updated with concrete unit numbers.
- No code has been written or modified.

**Change list — file-by-file assessment:**

| File | Status | Summary |
|------|--------|---------|
| `model/configuration.py` | Modify | Add MoSA-specific params; fix stale "Llama 3" docstring references; update model_type |
| `model/rope.py` | Modify | Add `rotate_fraction` support; existing position-gather logic already handles arbitrary indices |
| `model/attention.py` | Replace | Entire new MoSA hybrid attention layer; GQA module does not transfer |
| `model/decoder_layer.py` | Modify | Import new attention class; forward signature may need minor adaptation |
| `model/model.py` | Trivial | Class references already renamed; logic transfers unchanged |
| `model/huggingface.py` | Modify | Causal mask threading and cache handling need rethinking for MoSA |
| `model/mlp.py` | None | Transfers unchanged |
| `model/type_aliases.py` | None | Transfers unchanged |
| `model/__init__.py` | None | Transfers unchanged |
| `tokenizer.py` | None | Transfers unchanged |
| `upload_to_hub.py` | Trivial | Class names already renamed; content updates deferred to Unit 7 |

**Detail — configuration.py:**
Needs five new parameters for MoSA: head count split (dense vs sparse), sparsity denominator,
first-token forcing flag, and RoPE rotate fraction. Exact names and defaults are an open decision
(resolved at start of Unit 2). Existing params transfer. Stale "Llama 3" text in docstrings needs
updating. `model_type` should be simplified from `"mosa_baseline"` to `"mosa"`.

**Detail — rope.py:**
The existing `RotaryEmbedding.forward()` already gathers cos/sin at arbitrary `position_ids`, so
index-aware RoPE for sparse heads works without structural change. The only new requirement is
`rotate_fraction`: only a configurable fraction of head dimensions should receive rotation; the
remainder pass through unchanged. This requires a new parameter and a small change to the apply
step. Whether this is a method parameter or a constructor parameter is decided at Unit 4.

**Detail — attention.py:**
Full replacement. The MoSA hybrid layer:
- Dense heads: standard causal MHA/GQA (open decision: which — Unit 3)
- Sparse MoSA heads: per-head router (Linear → sigmoid), top-k selection, optional first-token
  forcing, gather from sequence, per-head Q/K/V projections, index-aware RoPE, structurally-derived
  causal mask (from original positions), SDPA, router-score gating of output, scatter back
- Layer output: sum of all head outputs
The file will be substantially new. Tests for this unit are entirely new.

**Detail — decoder_layer.py:**
Currently imports `GroupedQueryAttention`. Will import the new hybrid attention class instead. The
forward signature `(x, position_ids, cache, layer_idx, causal_mask)` may need revision depending
on what the new attention module requires — in particular, whether `causal_mask` is still relevant
for sparse heads (it is not — they build their own) or only for dense heads (potentially yes).
This is clarified during Unit 3 and resolved as a blocker if needed.

**Detail — huggingface.py:**
Currently builds a `causal_mask` for `use_cache=True` and threads it through the stack. With MoSA,
sparse heads ignore this mask (they derive causality from token indices). Dense heads may still use
it. Whether to keep, remove, or narrow the mask threading depends on what Unit 3 decides about
the dense head interface. KV-cache support (open decision) also lives here.

**Open decisions (must be resolved before the named unit begins):**

1. **Config param names and defaults** (Unit 2): What are the MoSA-specific config params called
   and what are their defaults? Names should follow the existing config conventions.

2. **Dense heads: GQA or MHA?** (Unit 3): The paper uses standard MHA. The baseline uses GQA.
   Dense heads in the hybrid layer — which do we use?

3. **Non-autoregressive routing** (Unit 3): During training the router sees the full sequence
   including future tokens; during inference it only sees past tokens. Do we accept this mismatch
   as-is (as the paper does), or add a mechanism to mask future tokens from the router?

4. **KV-cache support** (Unit 6): The reference implementation has no cache. Do we support
   `use_cache=True` at all? Options: disable entirely, support for dense heads only, or full
   support.

---

### Unit 2 — configuration.py

**What:** Add MoSA-specific parameters to `MosaConfig`. Fix stale "Llama 3" text in docstrings.
Update `model_type` to `"mosa"`. Resolve open decision 1 (param names and defaults) before
starting.

**Invariants this unit must satisfy:**
- All MoSA-specific parameters are present with correct types, defaults, and validation.
- All existing parameters from the Llama 3 baseline that remain relevant are present and unchanged.
- `model_type` is `"mosa"`.
- All parameters are documented with rationale consistent with the style of the existing config.
- `test_configuration.py` passes and accurately reflects the new parameters.

---

### Unit 3 — attention.py

**What:** Replace the GQA module with the MoSA hybrid attention layer. Resolve open decisions 2
(GQA/MHA for dense heads) and 3 (non-autoregressive routing) before starting. The sparse head
computation — router, gather, projection, index-aware RoPE, causal mask, attention, gating,
scatter — is entirely new. Dense heads are adapted from the existing attention module.
`test_attention.py` is substantially rewritten for the new component.

This unit is likely to surface blockers (e.g. rope.py needing rotate_fraction before attention
can be completed). Those are pushed onto the stack and resolved in order.

**Invariants this unit must satisfy:**
- The hybrid layer accepts `[B, T, h]` and returns `[B, T, h]`.
- Sparse heads attend only to their k selected tokens; causality is enforced by original positions.
- RoPE uses original sequence positions for sparse heads.
- Router gradients flow through the scalar gating (the gradient bridge through non-differentiable top-k).
- Dense and sparse paths are independently testable.
- All parameters are driven by config.

---

### Unit 4 — rope.py

**What:** Add `rotate_fraction` support to `RotaryEmbedding`. Only the specified fraction of head
dimensions receive rotation; the rest pass through unchanged. This is required by MoSA sparse
heads. Whether this is a constructor or call-site parameter is decided at the start of this unit.

**Invariants this unit must satisfy:**
- Standard behaviour (rotate_fraction=1.0) is identical to the existing implementation.
- At rotate_fraction < 1.0, only the specified fraction of dimensions is rotated.
- `test_rope.py` covers both the full-rotation and partial-rotation cases.

---

### Unit 5 — decoder_layer.py

**What:** Update the import to use the new hybrid attention class. Adapt the forward signature if
Unit 3 changed what the attention module expects. Minimal changes only.

**Invariants this unit must satisfy:**
- The decoder layer correctly composes the new attention module with RMSNorm and MLP.
- `test_decoder_layer.py` passes.

---

### Unit 6 — huggingface.py

**What:** Resolve open decision 4 (KV-cache support). Adapt causal mask construction and threading
to match what the hybrid attention layer actually needs after Unit 3. Minimal changes beyond what
is required.

**Invariants this unit must satisfy:**
- Forward pass contract is correct and documented.
- Cache behaviour (whether supported and how) is tested and documented.
- Causal masking is correct for all code paths.

---

### Unit 7 — upload_to_hub.py

**What:** Adapt the upload script for the MoSA model type. Update class names, model type string,
and model card content. Register `MosaConfig` and `MosaForCausalLM` with the AutoClass API and
push all model files to the Hub.

**Invariants this unit must satisfy:**
- A freshly instantiated model can be round-tripped: upload → `from_pretrained` → forward pass.
- The model card accurately describes the MoSA architecture.
- No weights are uploaded. No checkpoint is assumed.

---

### Unit 8 — Documentation

**What:** Write `documentation.md` covering design decisions, deviations from the paper, and
limitations. Update `README.md` with accurate architectural details. Record every open decision
resolved during implementation and the rationale for each.

**Invariants this unit must satisfy:**
- Every open decision resolved during implementation is recorded with its rationale.
- Limitations are documented explicitly.
- The model card accurately describes the MoSA variant.

---

### Unit 9 — End-to-End Tests

**What:** Full-stack smoke tests: instantiate from config, run a training step, verify loss
decreases. Mirror the Llama 3 end-to-end test structure. Include network tests for the Hub
round-trip.

**Invariants this unit must satisfy:**
- The model can be instantiated from a config, run a forward pass, compute loss, and backpropagate
  without error.
- Network tests verify the Hub round-trip.

---

### Unit 10 — Audit

**What:** Review every file in `src/mosa/` against the invariants in `job.md`. Verify no
hardcoded values, no missing documentation, no gaps between tests and intent. Apply the
close-the-testing-gap rule to any defect found.

**Invariants this unit must satisfy:**
- Every invariant in job.md is satisfied and has a corresponding test.
- No file has hardcoded architectural parameters.
- All documentation standards are met.
