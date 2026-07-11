# Krylov Methods

`rustebra` provides iterative eigenvalue methods in `rustebra::krylov`: power iteration for
the dominant (largest-magnitude) eigenvalue, and inverse power iteration for the eigenvalue
nearest an arbitrary shift. Unlike the direct decompositions in
[Decompositions](../06-decompositions/README.md), these refine an estimate over many
iterations and can fail to converge within a given budget, in addition to the usual
dimension and non-finite-value failure modes.
