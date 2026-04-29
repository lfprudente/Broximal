# Broximal Point Method

This repository contains the MATLAB codes used in the numerical experiments of the paper on the Broximal method.

## Reference

This code accompanies the paper:

> F. Babu, O. P. Ferreira, L. F. Prudente, Jen-Chih Yao, and Xiaopeng Zhao
> Ball-proximal point method on Hadamard manifolds, 2026.

If you use this code in academic work, please cite the paper.

## Implemented methods

The repository includes implementations of the following methods:

- **Broximal-F**: Broximal method with fixed radius
- **Broximal-A**: Broximal method with adaptive radius
- **Broximal-P**: Broximal method with Polyak-type radius
- **Proximal**: inexact proximal point method
- **Gradient**: gradient method with Armijo line search

The ball-constrained subproblems are approximately solved by a projected gradient method with Armijo line search, and the proximal subproblems are approximately solved by a gradient method with Armijo line search.

## Files

All MATLAB scripts and functions are stored in the same folder.

Main files:

- **main.m**: runs the numerical experiments and generates the LaTeX output table
- **Broximal.m**: Broximal method
- **ProjectedGradient.m**: projected gradient solver for the ball-constrained subproblems
- **Proximal.m**: inexact proximal point method
- **GradientProx.m**: gradient solver for the proximal subproblems
- **Gradient.m**: gradient method for the original problem

## Running the code

Open MATLAB in the repository folder and run `main`.

If `print_data = true`, the script generates a LaTeX file with the numerical results.

## Problem setting

The experiments consider unconstrained strongly convex quadratic problems of the form

minimize f(x) = (1/2) x' A x,

where `A` is symmetric positive definite.

## Requirements

- MATLAB

## License

This project is licensed under the MIT License. For more details, see the `LICENSE` file included in this repository.