% normal_collapse_sim.m
% MATLAB equivalent of the provided Python simulation:
% simulate repeated sampling from N(mu, sigma) and compare ensemble-average std to theory

clear; close all; clc;

%% ---------------- Parameters (edit as desired) ----------------
mu_0 = 0;              % initial mean
sigma_0 = 1.0;         % initial standard deviation
n = 100;               % samples per generation (M in paper)
num_trajectories = 25; % number of independent trajectories
num_generations = 2500;% number of generations (t_max)
rng_seed = 42;         % RNG seed for reproducibility
useDarkBackground = false; % set true if you want dark figure background
outFile = 'normal_collapse_comparison_plot.png';
%% ----------------------------------------------------------------

rng(rng_seed); % set RNG seed

% Preallocate storage: rows = trajectories, cols = generations
stds = zeros(num_trajectories, num_generations);
means = zeros(num_trajectories, num_generations);

% Setup a waitbar for progress (optional)
hWait = waitbar(0, 'Running simulation...', 'Name','Simulation Progress');

for traj = 1:num_trajectories
    mu = mu_0;
    sigma = sigma_0;
    for gen = 1:num_generations
        % generate n normal samples with current parameters
        samples = mu + sigma .* randn(n,1);   % vector n x 1
        
        % MLE estimates (sample mean and sample std with ddof=1)
        mu = mean(samples);         % sample mean
        sigma = std(samples);       % MATLAB default uses N-1 -> matches ddof=1 in numpy
        
        % store
        means(traj,gen) = mu;
        stds(traj,gen)  = sigma;
    end
    waitbar(traj/num_trajectories, hWait, sprintf('Trajectory %d / %d', traj, num_trajectories));
end

close(hWait);

% Ensemble average (over trajectories) for each generation
ensemble_average_stds = mean(stds, 1);   % 1 x num_generations

% Theoretical collapse function (vectorized)
generations = 0:(num_generations-1);  % match Python's arange
theoretical_stds = theoretical_collapse_rate(generations, n, sigma_0);

%% ---------------- Plotting ----------------
fig = figure('Units','normalized','Position',[0.1 0.12 0.7 0.6]);
if useDarkBackground
    set(gcf,'Color','k');
    ax = axes('Parent',fig,'Color','k','XColor','w','YColor','w');
else
    ax = axes('Parent',fig);
end
hold(ax,'on');

% plot ensemble average (note we map generations(1) -> index 1)
plot(ax, generations, ensemble_average_stds, 'b-', 'LineWidth', 2, ...
    'DisplayName', sprintf('Simulation Average (%d trajectories)', num_trajectories));
plot(ax, generations, theoretical_stds, 'r--', 'LineWidth', 2, 'DisplayName', 'Theoretical collapse');

xlabel(ax, 'Generation (t)', 'FontSize', 12, 'Color', get(ax,'XColor'));
ylabel(ax, 'Standard deviation (\sigma_t)', 'FontSize', 12, 'Color', get(ax,'YColor'));
title(ax, 'Gaussian Model Collapse: Simulation vs Theory', 'FontSize', 14, 'Color', get(ax,'XColor'));
grid(ax,'on');
legend(ax, 'Location', 'northeast');
xlim(ax, [0 generations(end)]);
% set a y-limit that gives some visualization headroom
ylim_upper = max(max(ensemble_average_stds), max(theoretical_stds)) * 1.1;
ylim(ax, [0, ylim_upper]);

set(gca,'FontSize',11);

% Save figure
% Make sure dark background saved correctly if requested
if useDarkBackground
    set(gcf,'InvertHardcopy','off');
end
print(fig, outFile, '-dpng', '-r150');
fprintf('Saved figure to %s\n', fullfile(pwd, outFile));

%% ---------------- Functions ----------------
function th = theoretical_collapse_rate(t, n, sigma_0)
    % Matches the Python function:
    % sigma_0 * sqrt(1 / (1 + t / n)) - 0.00008*t
    % t may be vector
    th = sigma_0 .* sqrt(1 ./ (1 + (t ./ n))) - 0.00008 .* t;
end
