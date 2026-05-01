# Specification: advanced-transformers-lib — SHRAM Research Baseline

## Purpose

This document specifies the implementation of the SHRAM (Sparse Hybrid Token Routed Attention Mixture) architecture as the second model in `advanced-transformers-lib`. It is a sibling to the Llama 3 baseline under `src/shram/` and `tests/shram/`, sharing no source files with the Llama 3 implementation.

SHRAM is a Llama-style decoder-only transformer in which every attention layer is replaced with the SHRAM hybrid attention layer — a combination of a local sliding-window causal attention path and a sparse token-routed MoSRAH path. All other components follow the Llama 3 baseline exactly except where the SHRAM architecture requires adaptation. The implementation meets the standards of this library: all parameters through config, fully documented, fully tested, original synthesis.

The library is a repository, not a PyPI package. Distribution is via HuggingFace Hub.

**What this document is.** This specification defines what the correct implementation must be. It is the instrument by which correctness is defined and maintained across sessions and collaborators. Code that runs is not necessarily correct. Code that passes tests is not necessarily correct. Correct means every invariant in this document holds and every test accurately enforces the invariants it claims to enforce. An implementation that satisfies this document is the goal. Code produced without satisfying this document is waste, regardless of how functional it appears.

---

## Governing Principle

Hours of LLM work rejected due to failing to follow governing principles: 7.4
Total sessions completely rolled back: 7
**These rules are serious**

This is a research baseline, not a software project. That distinction determines everything about how this work is done.

Software succeeds when it works. A research baseline succeeds when it can be *trusted*. These are different standards with different failure modes. Code that runs, produces outputs, and passes tests is not necessarily correct. Code whose correctness cannot be verified is not a step toward a research baseline — it is a different thing entirely, and it must be discarded and rebuilt. This has already happened on this project. It is not a warning; it is a description of how this work operates.

The governing objective is correctness. Productivity is not the objective. Producing artifacts is not the objective. These may occur as a consequence of working correctly, but pursuing them at the expense of correctness produces work that looks like progress and is not. The cost is paid all at once when the work is thrown out.

This is especially important because capable implementers — human or otherwise — feel strong pressure toward "working." It is fast, legible, and feels productive. "Verified correct" is slower and sometimes requires stopping to say "I don't know yet." That pressure toward apparent productivity is the primary failure mode on this project and must be actively resisted every time it appears. The moment implementation begins without a clear answer to "what does correct mean here," the work is already at risk.

The measure of success at every step is not "does it run" and not "do the tests pass." It is: do the tests accurately enforce the invariants, and do the invariants correctly capture what the system must be? If the answer to any part of that is no, the step is not done. A passing test suite built on wrong invariants is not verification — it is false confidence, and it is worse than no tests because it conceals the problem.

Work from this document. Reason from invariants. Stop and ask when uncertain. Those are not constraints on getting things done — they are how correct things get done.

Trust nothing. Avoid ever handling ambiguity by choosing to support multiple outcomes with fallback paths. This may silently hide config errors that torpedo the entire study. All team members must agree on current project state before moving onto a next step. Sometimes, this means you will be ready to implement but must answer questions and ask permission to proceed first. This catches mistakes.

> **Work is not accepted unless it can be proven correct. Correct means every decision traces to a principle in this document — before work begins as a stated commitment, and after as a demonstrated record. A plan that cannot show this connection before execution is not approvable. Work that cannot show it after is not complete. This is not abstract: multiple days of work on this project have already been discarded because this standard was not maintained. It will happen again before the work is accepted rather than after.

---

## Intended Use

The primary interface is HuggingFace AutoClass. A researcher pulls the architecture from the Hub and instantiates a freshly initialized model from a config — no pretrained weights involved. Config parameters can be overridden via kwargs at instantiation time in the standard HuggingFace fashion.

```python
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# Pull config from Hub and override parameters as needed
config = AutoConfig.from_pretrained(
    "your-namespace/advanced-transformers-lib",
    trust_remote_code=True,
    num_hidden_layers=12,  # example override; SHRAM-specific params follow the same pattern
)

# Instantiate with fresh random weights — no checkpoint required
model = AutoModelForCausalLM.from_config(config)

# Load tokenizer from Hub
tokenizer = AutoTokenizer.from_pretrained("your-namespace/advanced-transformers-lib")

# Save and reload a checkpoint
model.save_pretrained("./checkpoint")
model = AutoModelForCausalLM.from_pretrained("./checkpoint", trust_remote_code=True)
```

Note: `trust_remote_code=True` is required for custom architectures and must be documented prominently in the library README. The library does not own the training loop or data pipeline. These are the responsibility of the consumer.

---

## Architecture

**Success invariants — the implementation is correct when all of the following hold:**

- Every decoder layer uses the SHRAM hybrid attention layer. No layer uses standard self-attention unchanged.

- The SHRAM layer computes H(x) = hl(x) + hs(x), where hl is a local sliding-window causal attention path and hs is the MoSRAH sparse routed path. These paths are independent components with independent parameters; their outputs are summed.

- The local path hl uses a flash attention formulation in a window-size mode — the kind where a window parameter (e.g., `window_size=(w, 0)`) is passed to the kernel and masking is handled internally by the implementation. A manually constructed sliding-window boolean mask passed as `attn_mask` is not an acceptable substitute and is architecturally incorrect.

- The sparse path hs (MoSRAH) uses token-choice routing: each input token selects exactly K of L available attention heads. L and K are configurable parameters expressed through config.

- Routing produces two distinct outputs that must not be confused:
  - Selection indices I: which K heads each token selected, determined by Top-K applied to biased routing scores (unbiased scores + learned expert bias b).
  - Routing probabilities P: the weights used for the output weighted reduction, gathered from the *unbiased* routing scores over the K selected heads and renormalized so they sum to 1 per token. The bias b must not influence P.

  Confusing biased and unbiased scores corrupts the gradient path from the output back to the router weights, breaking training in a way that may not surface immediately.

- Expert packing converts the token-choice routing result into expert-choice layout, placing each head's assigned tokens contiguously. This conversion uses a stable sort permutation keyed on head indices. Stable sort is a correctness requirement: it preserves causal ordering of tokens within each head's packed representation, which is the foundation on which the triangular causal mask inside BEA is correct. Unstable sort silently violates causality.

- Expert packing produces two auxiliary outputs that must be propagated through the forward pass:
  - Position tensor J: the original sequence position of each packed token, required by RoPE inside BEA.
  - Boolean mask M with exactly B×N×K true entries: identifies real (non-padded) positions within the padded packed tensor, required for BEA attention masking and expert unpacking.

- Bottlenecked Ensemble Attention (BEA) is applied to the packed expert-choice tensor. Each of the L heads has independent Q, K, V, O projection parameters at bottleneck width u = d/K. RoPE is applied using J. Causality is enforced by a standard triangular mask over packed positions. Padded positions must not contribute to attention; M must gate them out.

- RoPE applied within BEA must support two modes, both configurable via config:
  - Original-sequence-position mode: J carries the original token positions (0..N-1) from the full sequence.
  - Local-packed-position mode: J carries positions 0, 1, 2, ... within each head's packed slot.

  The experimentally correct mode is undetermined; both must be implemented and selectable via config.

- Expert unpacking applies the inverse permutation to restore token-choice ordering, producing ỹ ∈ R^{B×N×K×d}.

- The MoSRAH output is the weighted sum o = Σ_k ỹ_k P_k, where P are the unbiased renormalized routing probabilities. This weighted sum is the only gradient path from the output back to the router — if P is wrong, the router does not train.

- Load balancing uses the DeepSeek auxiliary-loss-free biasing strategy. The load-balance loss is implemented via a custom gradient operator whose backward pass emits updates to the expert bias b through the main optimizer, not through a standalone update step. The load-balance loss must appear in the model's forward output so the training loop can scale and apply it. The weight applied to this loss is the consumer's responsibility.

- All architectural parameters — L, K, u, window size, RoPE mode, and all parameters inherited from the Llama 3 baseline — are expressed through config. Nothing is hardcoded. This invariant holds regardless of whether a hardcoded value would happen to be numerically correct.

- The implementation is compatible with the HuggingFace AutoClass interface as defined in the Intended Use section.

- The implementation is compatible with the HuggingFace Cache protocol. `use_cache=True` is supported. The cache accumulates keys and values per head across forward passes; adding a token extends the per-head K and V tensors for that token's K selected heads. The precise mechanism must be derived from the paper and the architecture's routing structure, not assumed from the Llama 3 baseline.

- Every architectural decision is grounded in the paper. Where the paper specifies behavior, the implementation follows it. Where the paper is silent or incomplete, that gap is a decision point to surface — not a license to assume, infer, or copy from another source. An implementation decision that cannot be traced to a statement in the paper or an explicit user-approved resolution is not a verified decision.

- Components other than the attention layer are expected to transfer from the Llama 3 baseline with little or no modification. The planner must verify each component for compatibility and surface any required adaptations. An assumed transfer that has not been verified is not verified.

### Preliminary Context and Avenues of Attack

The following is provided as orientation for the planner. It is not authoritative and must not be treated as design decisions or specifications. Every point is a hypothesis to verify, not an instruction to follow.

- **The hybrid layer as a drop-in.** SHRAM likely replaces the existing attention module in the decoder layer with the same `[B, T, h] → [B, T, h]` interface. Whether the surrounding decoder layer structure requires any changes is to be verified.

- **Packing as the central complexity.** The expert packing and unpacking process is likely the most intricate component of this architecture: stable sort permutation, variable-length per-head token counts, zero-padding to maximum head length, and a mask that must be threaded through BEA and unpacking. This component is likely to surface blockers and demands careful invariant-first planning.

- **Custom gradient operator as novel territory.** The load balancing update requires a custom backward pass: the forward emits a scalar loss, but the backward writes gradients to the expert bias b rather than to the loss inputs. This is architecturally novel relative to the Llama 3 baseline and will require careful design as its own unit.

- **Causality as a chain, not a point.** Causal correctness in MoSRAH flows through two linked invariants: stable sort preserves causal ordering → triangular mask over packed positions enforces it. Either link can be broken without raising an error. Both must be tested together in a way that catches violations of the chain, not just violations of each link in isolation.

- **KV-cache.** MoSRAH routing is dynamic and full-sequence-dependent. Whether incremental inference with a KV-cache is supported, and if so how, is an open question. The planner must surface this explicitly before any caching behavior is implemented or assumed.

- **Paper gaps.** The primary reference is a research draft. Several sections are marked [TODO] and at least one architectural value (RoPE mode) is marked [Version] and is experimentally undetermined. Do not fill gaps silently — any gap encountered during planning or implementation is a decision point to surface.

---

## File Layout

The repository structure is:

```
advanced-transformers-lib/
├── src/
│   ├── llama3/
│   │   └── ...         (existing — do not modify)
│   └── shram/
│       └── ...
└── tests/
    ├── llama3/
    │   └── ...         (existing — do not modify)
    └── shram/
        └── ...
```

**File granularity policy:** One file per major responsibility. Tests mirror the src structure exactly — each src file has a corresponding test file. The precise file breakdown within `src/shram/` and `tests/shram/` is decided during unit planning when the actual responsibilities are visible, not specified ahead of time. Process specifies when this decision is made.

Each model folder contains its own `upload_to_hub.py` script. See Uploading to Hub for its responsibilities.

---

## Uploading to Hub

Each model folder contains an `upload_to_hub.py` script. This script is responsible for:

- Registering the model and config with the HuggingFace AutoClass API
- Pushing the config and modeling code files to the Hub
- Uploading the tokenizer to the Hub repository
- Generating and pushing a model card populated with the correct architectural details

The upload script never uploads weights and never assumes a pretrained checkpoint exists. Its sole job is to make the architecture and tokenizer available on the Hub so a researcher can instantiate a freshly initialized model via `from_config`.

If relevant, model cards are generated programmatically. Large text descriptions are stored as data, and if it is resolvable at upload time it is included as well.

---

## Reference Sources

The following sources are provided as conditioning context to inform an original synthesis. They are not a copy hierarchy and must not be treated as one. The coder must read and understand these sources and produce an original implementation informed by all of them.

**1. "An Examination of Sparse Attention for Long Context Purposes" (primary)**
Located at the repository root: `An_Examination_of_Sparse_Attention_for_Long_Context_Purposes.pdf`
Use for: the definitive architectural specification of SHRAM and MoSRAH — routing, expert packing and unpacking, BEA, load balancing, RoPE treatment, and transformer structure.
Limitation: This is a research draft. Several sections are marked [TODO] and at least one architectural value (RoPE mode) is marked [Version] and experimentally undetermined. Do not treat incomplete sections as specifications. Do not fill gaps silently — surface them as decision points.

**2. Llama 3 baseline (copied into this variant)**
The starting point for `src/shram/` is a copy of `src/llama3/` and its tests. This copy is the coder's working baseline — it is owned and modified freely. The copy is the reference; the original `src/llama3/` is not imported and need not be consulted further.
Use for: interface conventions, config schema patterns, HuggingFace integration patterns, and the code quality bar already established for this library.

**Synthesis requirements:**
- The SHRAM architecture must be an original synthesis informed by the reference paper. Wholesale copying is not acceptable.
- The Llama 3 copy sets the style and quality bar. All new and modified code must meet or exceed it.
- Where sources leave genuine design choices open, the coder makes a decision, implements it, and reports it explicitly for user review.
- Where the coder is uncertain about a decision, work stops and the user is asked. Do not resolve uncertainty silently.

---

## Code Quality

Clean, well-structured code is an unconditional requirement across the entire codebase. This is a first-class standard, not a means to an end, and applies without exception to every file. Code which cannot be trusted is wrong by definition.

**Structure:**
- All architectural parameters expressed through config, never hardcoded
- Clear separation of concerns — one responsibility per file
- Type hints on all function and method signatures
- No dead code
- Placeholders must raise `NotImplementedError`, never pass silently
- Code strives to follow best practices such as single responsibility

**Documentation and commenting:**
- All classes must have docstrings
- All public methods must have docstrings
- Private methods must have docstrings when the logic is not a single clear operation
- Code should be self-documenting wherever possible through clear naming and structure
- When code cannot be made clearly self-documenting through variable names and structure alone, it must be documented with block comments explaining what the block achieves and why this approach was chosen
- Do not document line by line narrating what the code does; document at the block level explaining what it accomplishes and why
- Skipping documentation on non-self-documenting code and writing useless line-by-line narration are both failure modes of equal severity

---

## Testing

A codebase that works but cannot be verified has no value for research. Only a verified implementation can be used to draw scientific conclusions. This is the governing philosophy: verified-but-imperfect is more valuable than working-but-unverified, because only the former can be trusted as a research baseline.

Tests are first-class artifacts of this project. They are written alongside the implementation, not appended afterward. A component without passing tests is not complete, regardless of how correct it appears.

**Requirements:**
- Each src file has a corresponding test file mirroring the src structure under `tests/shram/`
- Unit tests verify each component's correctness in isolation
- Integration tests verify that assembled combinations of components behave correctly together, scoped to the units involved — not the full model end to end unless that is the unit being verified
- Placeholders can fail and their failure is allowed, but must raise `NotImplementedError`
- Integration tests do not replicate the testing functionality and level of detail of unit tests; they verify that pieces work together, not that subunits were unit tested correctly
- When making changes to support new functionality for other work units, tests must be updated to accurately reflect the new correct behavior before the unit is considered verified. Passing tests that no longer reflect intent are not verification — they are false confidence.
- The coder must ask the user when uncertain how to correctly test a given component. A bad test is worse than no test.

---

## Process

**The governing objective is correctness, not code production.** These are not the same thing. An implementation that runs, produces outputs, and passes tests is not necessarily correct. An implementation that satisfies every invariant in this specification and has tests that accurately enforce those invariants is correct. Work that does not meet this standard is not progress — it is waste, regardless of how much effort produced it. This project has already discarded multiple days of work because process was not followed and the results could not be trusted. The process rules below exist to prevent that outcome. Deviating without explicit user approval is not acceptable.

**The unit invariant principle.** Each unit of work begins by stating the invariants it must satisfy — what correct behavior means for that unit — before any implementation is considered. Implementation follows from invariants; invariants do not follow from implementation. A plan that specifies implementation steps without first establishing what correctness means for each step is a guess, not a plan. When a detail changes, a plan built on invariants survives; a plan built on implementation steps collapses.

**1. Plan first.**
Before writing any code, produce a complete plan covering: module structure, implementation order, and known decision points requiring user input. Submit the plan for user review and do not proceed until approved.

**1a. Agree before acting.**
Review and implementation are distinct scopes. Transitioning from review or discussion into code requires explicit agreement from both parties. There is no way for correct modifications to the codebase to be made without an agreed scope — acting outside of one means all results, human or AI, are wrong by definition, regardless of whether the code appears correct in isolation. A process violation here does not produce questionable work; it produces invalid work, and that work is rolled back without review. Both parties must reach explicit consensus before any edit is made.

Part of that consensus — and what the implementer should actively seek — is the scope of autonomy: a clear statement of what can be determined independently and what cannot. One valid answer is "you may figure this out yourself," but the critical output is knowing what lies outside that grant. This is isomorphic to clarifying the specification before work begins. Without a defined scope of autonomy there is no basis for correct implementation decisions, and the implementer must stop and ask rather than assume.

**2. Keep the plan current.**
Update the plan and status continuously as work proceeds. The plan must reflect actual current state at all times so that work can be resumed after a session interruption without loss.

**3. Define a unit of work.**
A unit of work is a major responsibility. A correctly scoped unit is one the coder believes it can complete without mistakes at 95% confidence; placeholders are allowed but must be tested and failing. Upon completion it becomes a verified black box: the rest of the system depends on it without needing to re-examine its internals. Verification is what licenses the abstraction. A unit is not complete until its tests accurately describe its intended behavior and pass.

**4. Decide file granularity during unit planning.**
When beginning a unit of work, decide how that unit maps to files, applying the one-responsibility-per-file policy.

**5. Refactor as part of verification.**
Before a unit can be marked complete and verified, the coder must assess whether the files involved respect the one-responsibility policy. If a file covers multiple distinct responsibilities, it must be analyzed before the unit is considered done; only if it cannot be sanely split may it stay merged. If a potential split would produce a file with no independent reason to exist, do not split it. Refactoring is part of done, not optional cleanup.

**6. Handle blockers with a stack.**
When a blocker arises requiring work outside the current unit's scope, treat it as a new unit of work pushed onto a stack. Complete and verify the blocker unit — including updated tests and resolving placeholders — before returning to the interrupted unit. Only make the changes needed for compatibility; do not expand scope. When the blocker is resolved, pop the stack and resume.

**7. One unit at a time.**
Always resolve one unit at a time. Never begin a new unit while the current one is unverified.

**8. Surface decisions.**
Autonomous decisions are permitted but must be reported to the user for review. Uncertain decisions must be escalated before proceeding. Do not resolve ambiguity silently.

**9. Close the testing gap.**
When a defect is found — in audit or anywhere else — the resolution is not complete until two questions are answered: (1) what invariant did the existing tests fail to enforce, and (2) what test change closes that gap? The fix and the test correction are a single unit of work.

**10. Surface paper↔implementation conflicts.**
The primary reference is a research draft with incomplete sections and experimentally undetermined values. Where the paper is silent, contradictory, or leaves a value undetermined, this is a decision point — not a license to fill the gap autonomously. Surface it, resolve it explicitly with the user, and record the resolution in the plan before proceeding. Silent gap-filling is a defect.

### Writing Standards for Plan Units

Each plan unit must be written as a certification artifact, not merely a task description. A correct unit records: (1) the unit's responsibility, (2) why this unit is the globally correct action for the project at this abstraction boundary, including any relevant rejected alternatives, (3) the invariants that define correct behavior, (4) the tests that certify those invariants, and (5) a preliminary implementation strategy.

Invariants may state only externally meaningful truth conditions, contractual obligations, and other facts that must be true for the unit to be correct. They must not contain implementation decisions such as storage layout, helper names, algorithms, inheritance choices, or other "how to do it" details unless those details are themselves part of the required external contract.

The certification/rationale section must not read like a local patch note. It must explain why this unit, at this boundary, is the most correct action for the overall project, and why the main viable alternatives are less correct. Philosophy and engineering argument are not separate sections of thought: the rationale must embody the philosophy in the concrete engineering case to connect back to the program.

The preliminary implementation strategy is the place for preferred approaches, provisional design choices, and decision contracts. It may say things like "if X remains true, Y must be preferred" or "if the ground situation reveals Z, this approach must be reconsidered." It must clearly distinguish these from invariants and preserve the ability to adapt to discovered facts without silently changing what correctness means.

