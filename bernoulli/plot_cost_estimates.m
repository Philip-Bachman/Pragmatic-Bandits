% Do plotting for cost estimate justification

fnames = {'res_cost_estimates_10x1.mat',...
    'res_cost_estimates_10x3.mat',...
    'res_cost_estimates_10x5.mat'};
subset_colors = [0.0 0.35 0.70];

% Plot figure for 10-armed bandits and various subset sizes
figure();
axes('FontSize', 14);
axis auto;
hold on;
for idx=1:numel(fnames),
    load(fnames{idx});
    c = subset_colors(idx);
    scatter(comp_costs_ucb, mean(conf_times_union,2), 'o',...
        'MarkerEdgeColor',[c c c],'LineWidth',2);
end
xlabel('Analytical Complexity Estimate','fontsize',14);
ylabel('Empirical Expected Complexity','fontsize',14);
title('Complexity Estimate Predictivity','fontsize',14);
    
    