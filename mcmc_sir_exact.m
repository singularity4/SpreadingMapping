function samples = mcmc_sir_exact(A, beta, gamma, source, num_samples, burn_in, thin)
% MCMC_SIR_EXACT  Rejection-free Gibbs sampler for SIR mapping (exact).
%
% Reference: Tolic, Kleineberg, Antulov-Fantulin (2018).
%   Simulating SIR processes on networks using weighted shortest paths.
%   Scientific Reports 8, 6562. https://arxiv.org/abs/1612.08629

[I, J] = find(A);
E = length(I);
N = size(A, 1);

Tau_node = exprnd(1/gamma, N, 1);
Rho = exprnd(1/beta, E, 1);
Tau = Tau_node(I);
weights = Rho .* (Rho <= Tau);
W = sparse(I, J, weights, N, N);

samples = zeros(num_samples, N);
total_steps = burn_in + num_samples * thin;
sample_idx = 0;

for step = 1:total_steps
    node = randi(N);
    Tau_node(node) = exprnd(1/gamma);

    edge_idx = find(I == node);
    for k = 1:length(edge_idx)
        e = edge_idx(k);
        Rho(e) = exprnd(1/beta);
        if Rho(e) <= Tau_node(node)
            W(I(e), J(e)) = Rho(e);
        else
            W(I(e), J(e)) = 0;
        end
    end

    if step > burn_in && mod(step - burn_in, thin) == 0
        sample_idx = sample_idx + 1;
        samples(sample_idx, :) = graphshortestpath(W, source);
    end
end

end
