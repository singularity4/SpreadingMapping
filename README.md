# SpreadingMapping

MATLAB code for simulating SIR processes on networks via weighted shortest paths.

**Reference:**
Tolić, D., Kleineberg, K-K., & Antulov-Fantulin, N. (2018).
Simulating SIR processes on networks using weighted shortest paths.
*Scientific Reports*, 8, 6562.
https://doi.org/10.1038/s41598-018-24648-w

## Files

- `SIR_mapping.m` — direct sampling, single realization
- `mcmc_sir_meanfield.m` — rejection-free Gibbs sampler, mean-field variant
- `mcmc_sir_exact.m` — rejection-free Gibbs sampler, exact variant with dynamical correlations
- `interventions.m` — time-critical vaccination strategies (random, hubs, temporal); reproduces Fig. 7

## Usage

```matlab
D = SIR_mapping(A, N, E, beta, gamma, source);
samples = mcmc_sir_meanfield(A, beta, gamma, source, num_samples, burn_in, thin);
samples = mcmc_sir_exact(A, beta, gamma, source, num_samples, burn_in, thin);
