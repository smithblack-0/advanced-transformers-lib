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

- [ ] Preliminary — copy llama3 to mosa, rename, verify baseline tests pass
- [ ] Unit 1 — configuration.py
- [ ] Unit 2 — rope.py (index-aware extension for sparse heads)
- [ ] Unit 3 — attention.py (MoSA hybrid attention layer)
- [ ] Unit 4 — decoder_layer.py
- [ ] Unit 5 — model.py
- [ ] Unit 6 — huggingface.py
- [ ] Unit 7 — upload_to_hub.py
- [ ] Unit 8 — Documentation
- [ ] Unit 9 — End-to-End Tests
- [ ] Unit 10 — Audit

---

## Status

**Current state:** Plan written. Not yet started.

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

## Repository Structure

```
advanced-transformers-lib/
├── src/
│   └── mosa/
│       ├── model/                  ← contents uploaded flat to Hub root
│       │   ├── __init__.py
│       │   ├── configuration.py    # Unit 1
│       │   ├── rope.py             # Unit 2
│       │   ├── attention.py        # Unit 3
│       │   ├── decoder_layer.py    # Unit 4
│       │   ├── model.py            # Unit 5
│       │   ├── huggingface.py      # Unit 6
│       │   ├── type_aliases.py
│       │   └── README.md
│       ├── upload_to_hub.py        # Unit 7
│       └── tokenizer.py
└── tests/
    └── mosa/
        ├── __init__.py
        ├── test_tokenizer.py
        └── model/
            ├── __init__.py
            ├── test_configuration.py
            ├── test_rope.py
            ├── test_attention.py
            ├── test_decoder_layer.py
            ├── test_model.py
            └── test_huggingface.py
```

This structure is the expected outcome of the preliminary copy-and-rename step. File granularity is
confirmed at the start of each unit and may be revised if responsibilities turn out larger or
smaller than anticipated.

**RMSNorm:** `torch.nn.RMSNorm` is used directly. No separate file or unit. As in the Llama 3
baseline, each point of use carries a comment explaining why RMSNorm was chosen.

---

## Open Decisions

These must be resolved and surfaced to the user before the affected unit begins. They are recorded
here so they are not forgotten and are not resolved silently.

**1. KV-cache support.**
The MoSA reference implementation has no cache support. The paper describes a theoretical cache
reduction benefit (sparse heads cache only k KV slots rather than T) but does not implement it.
Options range from disabling `use_cache` entirely to implementing partial cache support for dense
heads only. The correct answer depends on research intent and must be decided before Unit 6
(huggingface.py).

**2. Non-autoregressive routing.**
During training, the MoSA router scores all tokens over the full input sequence — including future
tokens. During autoregressive inference, only past tokens are available. This train/inference
mismatch is acknowledged in the paper and is unresolved in the reference implementation. Whether to
accept this as-is (as the paper does) or introduce a mechanism to mask future tokens during routing
must be decided before Unit 3 (attention.py).

**3. Dense heads: GQA or MHA.**
The MoSA paper uses standard multi-head attention for its dense heads. The Llama 3 baseline uses
Grouped Query Attention. Whether the dense heads in the hybrid layer use GQA (consistent with the
baseline) or MHA (consistent with the paper) must be decided before Unit 3.

**4. MoSA-specific config parameter names and defaults.**
The set of parameters controlling head counts, sparsity, routing behaviour, and RoPE adaptation
must be named and given sensible defaults before Unit 1 (configuration.py). Naming follows the
conventions already established in the Llama 3 config.

---

## Implementation Order

The order below is preferred, not fixed. Blockers are handled via the stack mechanism.

0. **Preliminary** — copy `src/llama3/` → `src/mosa/`, copy `tests/llama3/` → `tests/mosa/`;
   rename all classes and strings from `Llama3`/`llama3` to `Mosa`/`mosa`; verify all copied
   tests pass before touching any logic.
1. **configuration.py** — resolve open decision 4; add MoSA-specific parameters
2. **rope.py** — verify existing RoPE transfers; implement index-aware variant for sparse heads
3. **attention.py** — MoSA hybrid layer; resolve open decisions 2 and 3 before starting
4. **decoder_layer.py** — verify copy transfers; adapt to new attention signature if needed
5. **model.py** — verify copy transfers; adapt if needed
6. **huggingface.py** — resolve open decision 1; adapt mask threading and cache handling
7. **upload_to_hub.py** — adapt for mosa model type and class names
8. **Documentation** — documentation.md and README.md model card
9. **End-to-End Tests** — full stack training and generation smoke tests
10. **Audit** — review every file against invariants in job.md; close any gaps found

---

## HuggingFace AutoClass Protocol

Same protocol as the Llama 3 baseline. The `auto_map` in `config.json` must point to the correct
class names and files. `model_type` must be unique — `"mosa"` is the expected value. The upload
script pushes the `model/` directory contents flat to the Hub root alongside `config.json`.

---

## Units of Work

---

### Preliminary — Copy and Rename

**What:** Copy `src/llama3/` to `src/mosa/` and `tests/llama3/` to `tests/mosa/` using shell
commands. Rename all class names, module references, and string identifiers from `Llama3`/`llama3`
to `Mosa`/`mosa` throughout. Update `model_type` in configuration.py to `"mosa"`. Verify all
copied tests pass before any logic is changed.

**Why it exists:** The Llama 3 baseline is a known-good verified implementation. Starting from a
working copy rather than a blank slate eliminates mechanical errors in the scaffolding and lets the
units that follow focus exclusively on what is genuinely new.

**Invariants this preliminary step must satisfy:**
- `src/mosa/` and `tests/mosa/` exist and mirror the llama3 structure exactly, with all names
  updated.
- All copied tests pass against the copied (unmodified) implementation.
- `src/llama3/` and `tests/llama3/` are untouched.

---

### Unit 1 — configuration.py

**What:** Adapt `MosaConfig` from the copied Llama 3 config. Verify all existing parameters are
still correct. Add MoSA-specific parameters for: head count split (dense vs sparse), sparsity
control (tokens selected per head), routing behaviour (first-token forcing), and RoPE adaptation
fraction. Exact names and defaults to be decided and surfaced before this unit begins (open
decision 4).

**Invariants this unit must satisfy:**
- All parameters from the Llama 3 config that remain architecturally relevant are present.
- All MoSA-specific parameters are present with correct types, defaults, and validation.
- `model_type` is `"mosa"`.
- `auto_map` points to the correct MoSA class names.
- All parameters are documented with rationale in this plan.

---

### Unit 2 — rope.py

**What:** Verify that the standard RoPE implementation transfers unchanged for dense heads. Implement
an index-aware RoPE variant for sparse heads: given a set of original sequence position indices,
apply rotary encodings at those positions rather than at contiguous positions 0..k-1. The
`rotate_fraction` parameter (controlling what fraction of head dimensions receive rotation) must be
supported and configurable. Whether this is an extension of the existing module or a separate
module is a decision to make at the start of this unit.

**Invariants this unit must satisfy:**
- Standard RoPE (used by dense heads) produces identical output to the Llama 3 baseline for the
  same inputs.
- Index-aware RoPE produces correct rotations at the specified original sequence positions.
- A token gathered from position `i` in a sparse head receives the same rotation it would have
  received at position `i` in a dense head.
- `rotate_fraction` is respected: dimensions outside the rotated fraction are passed through
  unchanged.
- All behaviour is driven by config; no hardcoded values.

---

### Unit 3 — attention.py (MoSA hybrid layer)

**What:** The primary new component. Implements the full MoSA hybrid attention layer combining
dense heads and sparse MoSA heads. Open decisions 2 (non-autoregressive routing) and 3 (GQA vs
MHA for dense heads) must be resolved and surfaced before this unit begins.

The sparse head computation:
- Router: learned per-head per-token scorer producing a score in [0, 1] via sigmoid.
- Top-k selection: select the k highest-scoring tokens per head; k is derived from config.
- Optional first-token forcing: if configured, token 0 is always included.
- Gather: extract the k selected tokens from the full sequence.
- Projection: per-head Q, K, V projections over the k-token subset only.
- Index-aware RoPE: apply rotary encodings at original sequence positions.
- Causal mask: derived structurally from original positions of selected tokens — token i attends
  to token j only if j's original position ≤ i's original position.
- Attention: standard scaled dot-product attention over the k×k subset.
- Router gating: scale attention output by router scores before output projection.
- Output projection and scatter: project back to hidden dimension, scatter to full sequence.

Dense head computation follows the Llama 3 baseline. Layer output is the sum of all head outputs.

**Invariants this unit must satisfy:**
- The hybrid layer accepts `[B, T, h]` and returns `[B, T, h]`.
- Sparse heads attend only to the selected k tokens; no other tokens receive or contribute
  attention in those heads.
- Causality is enforced correctly: no sparse head can attend to a token at a later original
  position.
- RoPE uses original sequence positions, not local positions within the gathered subset.
- Router gradients flow correctly — the scalar gating by router scores is the gradient bridge
  through the non-differentiable top-k operation.
- All parameters (head counts, sparsity, routing, RoPE fraction) are driven by config.
- Dense heads and sparse heads are independently verifiable in tests.

---

### Unit 4 — decoder_layer.py

**What:** Verify the copied decoder layer wires correctly to the new attention module. Adapt the
forward signature if the MoSA hybrid attention layer requires different arguments than the Llama 3
attention module. No logic changes beyond what compatibility requires.

**Invariants this unit must satisfy:**
- The decoder layer produces correct output when given known inputs.
- The attention sub-layer, norm, and MLP are composed in the correct order.
- Any changes to the forward signature are minimal and justified.

---

### Unit 5 — model.py

**What:** Verify the copied model stack wires correctly to the new decoder layer. Adapt if needed.
No logic changes beyond what compatibility requires.

**Invariants this unit must satisfy:**
- The model stack correctly passes inputs through all decoder layers.
- `position_ids` and `causal_mask` threading (if still applicable after Unit 3 decisions) is
  correct.
- Hidden states from all layers are correctly collected when `output_hidden_states=True`.

---

### Unit 6 — huggingface.py

**What:** Adapt the HuggingFace wrapper for the MoSA variant. Resolve open decision 1 (KV-cache)
before starting. Adapt causal mask construction and threading to account for the fact that sparse
heads derive their own causality internally — the external mask may only be relevant for dense
heads, or may not be needed at all. The correct answer must be reasoned through and surfaced.

**Invariants this unit must satisfy:**
- The model can be instantiated via `AutoModelForCausalLM.from_config` and trained.
- Forward pass contract (inputs, outputs, error conditions) is documented and enforced.
- KV-cache behaviour (whether supported and how) is clearly documented and tested.
- Causal masking is correct for all code paths.

---

### Unit 7 — upload_to_hub.py

**What:** Adapt the upload script for the MoSA model type. Update class names, model type string,
and model card content. The script must register `MosaConfig` and `MosaForCausalLM` with the
AutoClass API and push all model files to the Hub.

**Invariants this unit must satisfy:**
- A freshly instantiated model can be round-tripped: upload → `from_pretrained` → forward pass.
- The model card accurately describes the MoSA architecture.
- No weights are uploaded. No checkpoint is assumed.

---

### Unit 8 — Documentation

**What:** Write `documentation.md` covering design decisions, deviations from the paper, and any
limitations. Update `README.md` (model card) with accurate architectural details. Document all open
decisions that were made during implementation and the rationale for each.

**Invariants this unit must satisfy:**
- Every open decision resolved during implementation is recorded with its rationale.
- Limitations (e.g. train/inference routing mismatch, if accepted as-is) are documented
  explicitly.
- The model card accurately describes the MoSA variant.

---

### Unit 9 — End-to-End Tests

**What:** Full-stack smoke tests: instantiate from config, run a training step, verify loss
decreases. Mirror the Llama 3 end-to-end test structure.

**Invariants this unit must satisfy:**
- The model can be instantiated from a config, run a forward pass, compute loss, and backpropagate
  without error.
- Network tests verify the Hub round-trip.

---

### Unit 10 — Audit

**What:** Review every file in `src/mosa/` against the invariants in `job.md`. Verify no
hardcoded values, no missing documentation, no gaps between tests and intent. Apply the close-the-
testing-gap rule to any defect found.

**Invariants this unit must satisfy:**
- Every invariant in job.md is satisfied and has a corresponding test.
- No file has hardcoded architectural parameters.
- All documentation standards are met.
