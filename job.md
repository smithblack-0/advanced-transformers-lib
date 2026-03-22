# Specification: advanced-transformers-lib — Llama 3 Research Baseline

## Purpose

This document specifies the implementation of a Llama 3-style decoder-only transformer as the first model in `advanced-transformers-lib` — a research library designed to hold multiple architectural formulations alongside one another. The library is intended to be forked and modified to test architectural variations in controlled empirical experiments.

This specification dispatches the implementation of the Llama 3 baseline only. The library structure is deliberately designed to accommodate future model variants as siblings to the baseline. The coder implements the baseline and the structure that supports it; nothing else.

The library is a repository, not a PyPI package. Distribution is via HuggingFace Hub.

---

## Intended Use

The primary interface is HuggingFace AutoClass. A researcher pulls the architecture from the Hub and instantiates a freshly initialized model from a config — no pretrained weights involved. Config parameters can be overridden via kwargs at instantiation time in the standard HuggingFace fashion.

```python
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# Pull config from Hub and override parameters as needed
config = AutoConfig.from_pretrained(
    "your-namespace/advanced-transformers-lib",
    trust_remote_code=True,
    num_hidden_layers=16,  # example override
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

The library implements a Llama 3-style decoder-only transformer with the following components:

- **Positional encoding:** Rotary Position Embeddings (RoPE)
- **Attention:** Grouped Query Attention (GQA)
- **Activation:** SwiGLU
- **Normalization:** RMSNorm
- **Tokenizer:** Llama 3 tokenizer, uploaded to the Hub alongside the model architecture

All architectural parameters must be expressed through config. Nothing may be hardcoded. The config must be capable of expressing the full range of model scales without assumption. Violation of this invariant is a defect regardless of whether the hardcoded value happens to be correct for the immediate use case.

---

## File Layout

The repository structure is:

```
advanced-transformers-lib/
├── src/
│   └── llama3/
│       └── ...
└── tests/
    └── llama3/
        └── ...
```

Future model variants sit as siblings to `llama3/` under both `src/` and `tests/`.

**File granularity policy:** One file per major responsibility. Tests mirror the src structure exactly — each src file has a corresponding test file. The precise file breakdown within `src/llama3/` and `tests/llama3/` is decided during unit planning when the actual responsibilities are visible, not specified ahead of time. Process specifies when this decision is made

Each model folder contains its own `upload_to_hub.py` script. See Uploading to Hub for its responsibilities.

---

## Uploading to Hub

Each model folder contains an `upload_to_hub.py` script. This script is responsible for:

- Registering the model and config with the HuggingFace AutoClass API
- Pushing the config and modeling code files to the Hub
- Uploading the Llama 3 tokenizer to the Hub repository
- Generating and pushing a model card populated with the correct architectural details

The upload script never uploads weights and never assumes a pretrained checkpoint exists. Its sole job is to make the architecture and tokenizer available on the Hub so a researcher can instantiate a freshly initialized model via `from_config`.

If relevant, model card are generated programmatically. Large text descriptions are stored as data, and if it is resolvable at upload time it is included as well.

---

## Reference Sources

The following sources are provided as conditioning context to inform an original synthesis. They are not a copy hierarchy and must not be treated as one. The coder must read and understand these sources and produce an original implementation informed by all of them.

**1. "The Llama 3 Herd of Models"**
`https://arxiv.org/abs/2407.21783`
Use for: architectural intent, motivation, and design decisions at the model level.
Limitation: papers routinely underspecify implementation details. Do not treat as a complete implementation reference.

**2. Meta Llama 3 Repository**
`https://github.com/meta-llama/llama3`
Use for: official reference point and structural intent.
Limitation: explicitly minimal — inference example only. Too thin to serve as a sole implementation reference; weight accordingly.

**3. Meta Llama Models Repository**
`https://github.com/meta-llama/llama-models`
Use for: actual model utilities, architecture code, and post-3.1 consolidated implementation details.
Limitation: may overlap with HuggingFace upload; assess comment and organizational quality before relying on structure.

**4. HuggingFace Transformers — LlamaForCausalLM**
`https://github.com/huggingface/transformers` (`LlamaForCausalLM`)
Use for: interface conventions this library must satisfy. Authoritative for the HuggingFace contract.
Limitation: HuggingFace uploaded code is often stripped of comments and organization. Treat as interface reference only, not as a structural or style reference.

**5. HuggingFace Model Card**
`https://huggingface.co/meta-llama/Llama-3.1-8B`
Use for: config schema reference and tokenizer conventions.

**Synthesis requirements:**
- The implementation must be an original synthesis informed by the above sources. Wholesale copying from any source is not acceptable and may constitute copyright infringement.
- None of the above sources can be trusted as a style reference. Code quality standards are defined in this specification and inherited from nowhere.
- Where sources are in tension or leave genuine design choices open, the coder makes a decision, implements it, and reports it explicitly for user review.
- Where the coder is uncertain about a decision, it must stop and ask before proceeding.

---

## Code Quality

Clean, well-structured code is an unconditional requirement across the entire codebase. This is a first-class standard, not a means to an end, and applies without exception to every file. Code which cannot be trusted is wrong by definition.

**Structure:**
- All architectural parameters expressed through config, never hardcoded
- Clear separation of concerns — one responsibility per file
- Type hints on all function and method signatures
- No dead code
- Placeholders must raise `NotImplementedError`, never pass silently
- Code strives to follow best pratices such as single responsibility.

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
- Each src file has a corresponding test file mirroring the src structure under `tests/llama3/`
- Unit tests verify each component's correctness in isolation
- Integration tests verify that assembled combinations of components behave correctly together, scoped to the units involved — not the full model end to end unless that is the unit being verified.
- Placeholders can fail and their failure are allowed, but should raise a NotImplementedError
- Integration tests do not replicate the testing functionality and level of details of unit tests; they need to test pieces work together, not the other subunits were unit tested correctly.
- When making changes to support new functionality for other work units, tests must be updated to accurately reflect the new correct behavior before the unit is considered verified. Passing tests that no longer reflect intent are not verification — they are false confidence.
- The coder must ask the user when uncertain how to correctly test a given component. A bad test is worse than no test.

---

## Process

The coder must follow this process. Deviating without explicit user approval is not acceptable.

**1. Plan first.**
Before writing any code, produce a complete plan covering: module structure, implementation order, testing strategy per component, and known decision points requiring user input. Submit the plan for user review and do not proceed until approved.

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