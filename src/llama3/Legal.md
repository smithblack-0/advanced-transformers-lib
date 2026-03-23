# Legal

## License

This codebase is released under the MIT license. See `LICENSE` at the repository root for the
full license text.

The rest of this document explains why the MIT license applies — specifically, why no third-party
copyright prevents it.

---

## Clean-Room Synthesis

The architecture implementation is a clean-room synthesis. No source code from Meta's Llama
repositories was copied, read, or used as a direct implementation reference by any party
involved in writing this code. This eliminates any possibility of copyright transfer from
the Llama codebase to this project.

The process was:

1. The human author specified the requirements and governed the process throughout (see `plan.md`).
2. An LLM instance researched the architecture from published sources — the Llama 3 paper, the
   Meta reference repositories used as structural context only, and the HuggingFace Transformers
   interface conventions — and produced an original implementation plan.
3. Separate LLM instances implemented each unit from that plan. Those instances were not shown
   raw Llama source code at any point.
4. The human author reviewed all significant decisions, approved each unit before the next began,
   and corrected the process wherever it deviated.

The full process record, including every decision point and human sign-off, is in `plan.md`.

---

## Tokenizer

The tokenizer is GPT-NeoX (`EleutherAI/gpt-neox-20b`), released under the Apache 2.0 license.
The tokenizer files in `model/` are reproduced here under the terms of that license.
