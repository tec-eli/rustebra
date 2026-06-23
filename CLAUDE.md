# CLAUDE.md

Read `adr/` before writing any code. The ADRs are the source of truth for how this project is
built; don't re-derive or guess at decisions already made there.

Work from `TODO.md`, one unchecked item at a time. Don't jump ahead to future phases.

## Rust best practices

- No `unwrap()`/`expect()`/`panic!()` in library code, except in tests — failures are
  reported through `Result`, per ADR 0004.
- No `.clone()` to dodge a borrow-checker error. If a clone seems necessary, stop and check
  whether the function signature should borrow instead.
- Public items get a doc-comment (`///`) with a compiling example, not just a one-line
  description.
- Prefer `impl Trait` / generic bounds over `Box<dyn Trait>` — this is a `no_std`-first
  crate, so trait objects are the exception, not the default.
- Run `cargo clippy -- -D warnings` as the actual bar, not just `cargo build` succeeding.
- Match existing naming before inventing new naming — check `adr/` and already-written code
  first.
- No new dependency without checking it supports `no_std` (or is gated behind `alloc`/a
  feature) and is justified in the commit message.

## Rules

- Code for the current task gets a proper module (its own file, in a folder once there's more
  than one related file) — don't dump everything in `lib.rs`. What's not allowed is creating
  empty modules/folders reserved for *future* tasks that haven't started yet.
- Within a module, split into a folder with sub-files once it mixes distinct
  responsibilities — trait definition, per-type implementation, and the algorithm behind a
  method are three different responsibilities and belong in three different files (e.g.
  `scalar/mod.rs` for the trait, `scalar/f32.rs` / `scalar/f64.rs` for implementations,
  `scalar/sqrt.rs` for the Newton-Raphson algorithm). Don't keep growing a single file just
  because it's "still about the same trait".
- Don't add code beyond what the current task requires (no speculative helpers, no unused
  abstractions, no "while I'm here" extras).
- Check the box in `TODO.md` and add an entry to `CHANGELOG.md` under `[Unreleased]`.
- Do not add any comments referencing ADR files.
- Never reference, cite, or summarize ADR numbers or ADR content anywhere in source code — not in comments, not in doc-comments, not in commit messages.

## Changelog

- Do NOT add thing that are just necessary actions like running test commands, updating the version, checking if it can be deployed.
- When doing prepare for release activities just add "Release metadata"