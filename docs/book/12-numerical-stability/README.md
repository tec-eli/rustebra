# Numerical Stability

This section documents two aspects of how `rustebra` behaves numerically: its policy on
`NaN`/`Inf` propagation, and the precision trade-offs behind its fixed-iteration
approximations for `sqrt`, `sin`, and `cos`.
