# Book

This book documents `rustebra`, a hybrid `no_std`/`alloc` linear algebra crate for Rust. It
starts with the foundations — scalars and vectors — and builds up through matrices, sparse
storage, decompositions, and Krylov subspace solvers.

Part I covers the foundations: the `Scalar` trait that algorithms in this crate are generic
over, and the `StaticVector`/`DynamicVector` types built on top of it. Later parts assume
familiarity with these building blocks.
