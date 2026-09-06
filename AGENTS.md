# AGENTS.md — synapse-tutor

Agent session entry point for the Synapse Tutor Rust XOR neural network demo.

## Start here

1. Read `README.md` for the architecture diagram and expected training output.
2. All logic is in `src/main.rs` (single-file crate) with inline `#[cfg(test)]` tests.
3. `Cargo.toml` pins Rust edition 2021 and MSRV 1.70.

## Project layout

```
Cargo.toml       Package manifest (single binary, depends on rand)
src/main.rs      NeuralNetwork struct, sigmoid, forward pass, backprop training, main demo, tests
```

## Development commands

```bash
# Run the XOR training demo
cargo run --release

# Run the test suite
cargo test

# Lint
cargo clippy --all-targets

# Format check
cargo fmt --check
```

## Key conventions

- Single-file crate with a `NeuralNetwork` implementing 2→4→1 MLP.
- Uses `rand::distributions::Uniform` for Xavier-like random initialization.
- Activation is sigmoid; training uses backpropagation on the four XOR samples.
- `clippy::needless_range_loop` is allowed for readability in the math-heavy loops.
- Tests assert sigmoid properties, forward output ranges, dimension preservation, and that training reduces error/learns XOR.

## Gotchas

- The crate contains a `.cargo/config.toml` that points to a non-existent `/.cargo-targets` directory. Remove or override it if local builds fail with "Read-only file system".
- Training is non-deterministic due to random init, so XOR learning is checked with a 0.1 tolerance threshold.
- `cargo test` runs both unit tests and includes the demo code paths; tests do not call `main()`.

## Quality gates

Run before committing:

```bash
cargo test
cargo clippy --all-targets
cargo fmt --check
```
