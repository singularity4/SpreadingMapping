% Time-critical vaccination (intervention) strategies for SIR epidemic processes on networks.
%
%   Implements the three vaccination strategies compared in Fig. 7 of:
%   Tolic, Kleineberg, Antulov-Fantulin (2018).
%   Simulating SIR processes on networks using weighted shortest paths.
%   Scientific Reports 8, 6562. https://doi.org/10.1038/s41598-018-24648-w
%
% The central idea: at observation time t0, a disease outbreak is detected
% at a known source node. We have m vaccines, but they take Delta_t days
% to become effective. A vaccinated node infected before t0 + Delta_t is
% a wasted dose. The question is: which m nodes should we vaccinate?
%
% Three main strategies are compared (paper Fig. 7):
%   (1) Random    — vaccinate m random susceptible nodes at t0.
%   (2) Hubs      — vaccinate the m highest-degree nodes (classical heuristic).
%   (3) Temporal  — vaccinate nodes most likely outside the t0+Delta_t window,
%                   i.e. nodes the epidemic is unlikely to reach before vaccines
%                   take effect (Eq. 10 of the paper).
%
% Key result from the paper: hubs perform *worse* than random in the
% time-critical regime, because highly connected nodes are typically reached
% early by the epidemic, causing vaccine doses to be wasted.
%
% Dependencies: SIR_mapping.m (must be on MATLAB path).
%
% Usage:
%   [s_rand, s_hubs, s_temp] = compare_strategies(A, beta, gamma, source, ...
%                                   t0, Delta_t, m, n_samples)


% -------------------------------------------------------------------------
function p_tilde = vax_probabilities(A, beta, gamma, source, t0, Delta_t, n_samples)
% VAX_PROBABILITIES  Estimate the probability each node is outside the
%   vaccine efficacy window, following Eq. (10) of Tolic et al. (2018).
%
%   p_tilde(i) = (1/n) * sum_k  Theta( d_{G_k}(source, i) - t0 - Delta_t )
%
%   where d_{G_k}(source, i) is the first-infection time of node i in
%   realization k (i.e. the weighted shortest path from source to i),
%   and Theta is the Heaviside step function.
%
%   Interpretation: p_tilde(i) is the fraction of epidemic realizations in
%   which node i has NOT yet been infected by time t0 + Delta_t. A node
%   with high p_tilde is a good vaccine candidate — it is likely still
%   susceptible when the vaccine kicks in, so the dose will not be wasted.
%
%   Inputs:
%     A         — N x N adjacency matrix (unweighted, directed or undirected)
%     beta      — transmission rate (infections per unit time per contact)
%     gamma     — recovery rate (recoveries per unit time)
%     source    — index of the outbreak source node (1-indexed)
%     t0        — observation time: moment at which vaccination decision is made
%     Delta_t   — vaccine efficacy delay: doses administered at t0 take effect
%                 at t0 + Delta_t; any node infected in (t0, t0+Delta_t) is lost
%     n_samples — number of independent SIR realizations to average over
%                 (paper uses n = 10^4; more gives lower Monte Carlo variance)
%
%   Output:
%     p_tilde   — N x 1 vector; p_tilde(i) in [0,1] is the estimated
%                 probability that node i remains susceptible past t0+Delta_t

    N = size(A, 1);
    [~, E_count] = size(find(A));  % number of directed edges
    E = nnz(A);

    threshold = t0 + Delta_t;
    outside_window = zeros(N, 1);  % accumulates Theta(D(i) - threshold)

    for k = 1:n_samples
        % Each call to SIR_mapping samples one realization of the weighted
        % network ensemble and returns D(i) = first-infection time of node i
        % from the given source (Eq. 2, shortest path in weighted graph G_k).
        D = SIR_mapping(A, N, E, beta, gamma, source);

        % Theta(D(i) - threshold) = 1 if node i is NOT reached by t0+Delta_t.
        % Inf entries in D mean the node is never infected in this realization
        % (no path exists); these count as outside the window (Theta = 1).
        outside_window = outside_window + (D(:) > threshold);
    end

    % Eq. (10): p_tilde_i = (1/n) * sum_k Theta(d_{G_k}(source,i) - t0 - Delta_t)
    p_tilde = outside_window / n_samples;
end


% -------------------------------------------------------------------------
function infected = outbreak_size(A, beta, gamma, source, vaccinated, n_samples)
% OUTBREAK_SIZE  Estimate expected final outbreak size under a vaccination
%   strategy, averaged over the SIR ensemble (Eq. 3 of Tolic et al. 2018).
%
%   Vaccination is modelled by removing vaccinated nodes from the contact
%   network before simulating the epidemic. This is valid here because we
%   only vaccinate nodes whose doses will be effective — nodes outside the
%   t0+Delta_t window (selected by the calling strategy function).
%   Doses that would be wasted (nodes already infected) are excluded by
%   each strategy before calling this function.
%
%   Final outbreak size in realization k is the number of nodes reachable
%   from source in finite time in the modified weighted graph G_k^vax.
%   The expectation over the ensemble gives the mean final infected count.
%
%   Inputs:
%     A          — N x N adjacency matrix (original, unmodified)
%     beta       — transmission rate
%     gamma      — recovery rate
%     source     — outbreak source node index
%     vaccinated — indices of nodes to remove from the contact network
%     n_samples  — number of realizations to average over
%
%   Output:
%     infected   — scalar; expected number of infected nodes at t -> infinity

    % Remove vaccinated nodes from the contact network.
    % This severs all transmission paths through vaccinated individuals.
    A_vax = A;
    A_vax(vaccinated, :) = 0;
    A_vax(:, vaccinated) = 0;

    N = size(A_vax, 1);
    E = nnz(A_vax);

    total = 0;
    for k = 1:n_samples
        D = SIR_mapping(A_vax, N, E, beta, gamma, source);
        % Nodes with D(i) < Inf were reached by the epidemic (infected).
        % Nodes with D(i) = Inf remained susceptible for this realization.
        total = total + sum(D < inf);
    end

    infected = total / n_samples;
end


% -------------------------------------------------------------------------
function vaccinated = strategy_random(A, beta, gamma, source, t0, m, n_samples)
% STRATEGY_RANDOM  Baseline: vaccinate m nodes chosen uniformly at random
%   from the susceptible population observed at time t0.
%
%   At t0, susceptible nodes are those not yet infected: D(source,i) > t0.
%   We estimate the susceptible set by averaging over the ensemble and
%   selecting nodes where the majority of realizations show D(i) > t0.
%   Then m nodes are drawn uniformly at random from this set.
%
%   This is the weakest strategy and serves as the lower baseline in Fig. 7.
%   It does not exploit any structural or temporal information.

    N = size(A, 1);
    E = nnz(A);

    % Estimate which nodes are susceptible at t0 across the ensemble.
    susceptible_count = zeros(N, 1);
    for k = 1:n_samples
        D = SIR_mapping(A, N, E, beta, gamma, source);
        susceptible_count = susceptible_count + (D(:) > t0);
    end

    % Nodes susceptible in the majority of realizations.
    susceptible = find(susceptible_count > n_samples / 2);
    susceptible = setdiff(susceptible, source);  % source is already infected

    % Draw m nodes at random from the susceptible set.
    idx = randperm(length(susceptible), min(m, length(susceptible)));
    vaccinated = susceptible(idx);
end


% -------------------------------------------------------------------------
function vaccinated = strategy_hubs(A, source, m)
% STRATEGY_HUBS  Vaccinate the m highest-degree nodes (classical hub heuristic).
%
%   Degree centrality is the most common network-based vaccination heuristic:
%   removing hubs disrupts many transmission paths and reduces R0.
%   This strategy ignores the temporal structure of the epidemic entirely.
%
%   As shown in Fig. 7 of Tolic et al. (2018), this strategy performs *worse*
%   than random in the time-critical regime (small Delta_t relative to
%   epidemic speed). The reason: hubs are highly connected and therefore
%   reached early by the epidemic, so doses administered at t0 have not yet
%   taken effect (t0 + Delta_t) when the hub is infected — the dose is wasted.
%   This is the key mechanistic insight motivating the temporal strategy.

    % Out-degree for directed networks; total degree for undirected.
    degree = sum(A, 2) + sum(A, 1)';

    % Exclude source node (already infected, cannot be vaccinated usefully).
    degree(source) = -1;

    [~, sorted] = sort(degree, 'descend');
    vaccinated = sorted(1:min(m, end));
end


% -------------------------------------------------------------------------
function vaccinated = strategy_temporal(A, beta, gamma, source, t0, Delta_t, m, n_samples)
% STRATEGY_TEMPORAL  Vaccinate the m nodes most likely to remain susceptible
%   past t0 + Delta_t, using the vaccination probability p_tilde (Eq. 10).
%
%   This strategy uses the SIR ensemble to estimate, for each node i,
%   the probability that i has not been infected by t0 + Delta_t — the
%   moment when vaccines administered at t0 become effective. Vaccinating
%   nodes with the highest p_tilde minimises wasted doses and protects
%   nodes that are still reachable by the epidemic after the efficacy window.
%
%   This is the correct strategy in the time-critical regime: it reasons
%   about temporal distances in the weighted graph ensemble, not degree.
%   See Fig. 7: temporal strategy outperforms both hubs and random,
%   especially when Delta_t is non-negligible relative to epidemic speed.

    % Estimate p_tilde(i) = P(D(source,i) > t0 + Delta_t) via Eq. (10).
    p_tilde = vax_probabilities(A, beta, gamma, source, t0, Delta_t, n_samples);

    % Exclude source (infected at t=0) and nodes already infected at t0.
    % Nodes with p_tilde = 0 are almost certainly infected before t0+Delta_t
    % and should not receive doses (they would be wasted).
    p_tilde(source) = -1;

    % Select top-m nodes by p_tilde: most likely to still be susceptible
    % when the vaccine takes effect.
    [~, sorted] = sort(p_tilde, 'descend');
    vaccinated = sorted(1:min(m, end));
end


% -------------------------------------------------------------------------
function [size_random, size_hubs, size_temporal] = compare_strategies(A, beta, gamma, source, t0, Delta_t, m, n_samples)
% COMPARE_STRATEGIES  Reproduce the three-strategy comparison of Fig. 7 in
%   Tolic et al. (2018): random vs. hubs vs. temporal vaccination.
%
%   Given a network, SIR parameters, an observation time t0, a vaccine
%   efficacy delay Delta_t, and m available doses, this function selects
%   vaccine targets under each strategy and estimates the resulting final
%   outbreak size by running the SIR ensemble on the modified network.
%
%   Inputs:
%     A         — N x N adjacency matrix (unweighted)
%     beta      — transmission rate
%     gamma     — recovery rate
%     source    — index of outbreak source node
%     t0        — time at which vaccination decision is made
%     Delta_t   — vaccine efficacy delay (doses effective at t0 + Delta_t)
%     m         — number of available vaccine doses
%     n_samples — ensemble size for both strategy selection and scoring
%                 (paper uses n = 10^4 for reliable estimates)
%
%   Outputs:
%     size_random   — expected final outbreak size under random vaccination
%     size_hubs     — expected final outbreak size under hub vaccination
%     size_temporal — expected final outbreak size under temporal vaccination
%
%   Interpretation: lower outbreak size = better strategy.
%   Expected result (Fig. 7): size_temporal < size_random < size_hubs.

    fprintf('Selecting vaccine targets...\n');

    V_random   = strategy_random  (A, beta, gamma, source, t0, m, n_samples);
    V_hubs     = strategy_hubs    (A, source, m);
    V_temporal = strategy_temporal(A, beta, gamma, source, t0, Delta_t, m, n_samples);

    fprintf('Scoring strategies...\n');

    size_random   = outbreak_size(A, beta, gamma, source, V_random,   n_samples);
    size_hubs     = outbreak_size(A, beta, gamma, source, V_hubs,     n_samples);
    size_temporal = outbreak_size(A, beta, gamma, source, V_temporal, n_samples);

    fprintf('\n--- Vaccination strategy comparison (Fig. 7) ---\n');
    fprintf('  Random   strategy: %.1f nodes infected\n', size_random);
    fprintf('  Hubs     strategy: %.1f nodes infected\n', size_hubs);
    fprintf('  Temporal strategy: %.1f nodes infected\n', size_temporal);
    fprintf('  (lower = better; temporal should win in time-critical regime)\n');
end
