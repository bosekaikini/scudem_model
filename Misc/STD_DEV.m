

clear; close all; clc;


mu_0 = 0;              
sigma_0 = 1.0;         
n = 100;               
num_trajectories = 25; 
num_generations = 2500;
rng_seed = 42;         
useDarkBackground = false; 
outFile = 'normal_collapse_comparison_plot.png';


rng(rng_seed);


stds = zeros(num_trajectories, num_generations);
means = zeros(num_trajectories, num_generations);


hWait = waitbar(0, 'Running simulation...', 'Name','Simulation Progress');

for traj = 1:num_trajectories
    mu = mu_0;
    sigma = sigma_0;
    for gen = 1:num_generations
        
        samples = mu + sigma .* randn(n,1);   % vector n x 1
        
        
        mu = mean(samples);         
        sigma = std(samples);       
        
        
        means(traj,gen) = mu;
        stds(traj,gen)  = sigma;
    end
    waitbar(traj/num_trajectories, hWait, sprintf('Trajectory %d / %d', traj, num_trajectories));
end

close(hWait);


ensemble_average_stds = mean(stds, 1);   


generations = 0:(num_generations-1);  % match Python's arange
theoretical_stds = theoretical_collapse_rate(generations, n, sigma_0);


fig = figure('Units','normalized','Position',[0.1 0.12 0.7 0.6]);
if useDarkBackground
    set(gcf,'Color','k');
    ax = axes('Parent',fig,'Color','k','XColor','w','YColor','w');
else
    ax = axes('Parent',fig);
end
hold(ax,'on');


plot(ax, generations, ensemble_average_stds, 'b-', 'LineWidth', 2, ...
    'DisplayName', sprintf('Simulation Average (%d trajectories)', num_trajectories));
plot(ax, generations, theoretical_stds, 'r--', 'LineWidth', 2, 'DisplayName', 'Theoretical collapse');

xlabel(ax, 'Generation (t)', 'FontSize', 12, 'Color', get(ax,'XColor'));
ylabel(ax, 'Standard deviation (\sigma_t)', 'FontSize', 12, 'Color', get(ax,'YColor'));
title(ax, 'Gaussian Model Collapse: Simulation vs Theory', 'FontSize', 14, 'Color', get(ax,'XColor'));
grid(ax,'on');
legend(ax, 'Location', 'northeast');
xlim(ax, [0 generations(end)]);

ylim_upper = max(max(ensemble_average_stds), max(theoretical_stds)) * 1.1;
ylim(ax, [0, ylim_upper]);

set(gca,'FontSize',11);


if useDarkBackground
    set(gcf,'InvertHardcopy','off');
end
print(fig, outFile, '-dpng', '-r150');
fprintf('Saved figure to %s\n', fullfile(pwd, outFile));


function th = theoretical_collapse_rate(t, n, sigma_0)
   

    th = sigma_0 .* sqrt(1 ./ (1 + (t ./ n))) - 0.00008 .* t;
end
