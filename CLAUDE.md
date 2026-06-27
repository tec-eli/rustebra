# CLAUDE.md

Read `docs/adr/` before writing any code. 

## Non-negotiable rules

- No `unwrap()`/`expect()`/`panic!()` in library code — use `Result` (ADR 0004).
- No `.clone()` to escape a borrow error — fix the signature.
- No `Box<dyn Trait>` by default — `no_std`-first means generics/`impl Trait` (ADR 0001).
- No new dependency without confirming `no_std` support.
- No speculative code — only what the current task requires.
- Lines ≤ 120 characters.
- Public items get a `///` doc-comment with a compiling `# Examples` block.
- New code gets its own module. Split into a folder when a module mixes distinct
  responsibilities (trait def / per-type impl / algorithm = three files).
- Never mention ADR numbers or content in source code, comments, or commit messages.
- Match existing naming — check `docs/adr/` and existing code first.

## After every task

- Do not log mechanical actions.
- Do not commit anything unless asked to do so.
