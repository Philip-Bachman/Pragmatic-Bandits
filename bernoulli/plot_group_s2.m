% Do plotting for comparing selection directed towards finding that the first
% arm of any given bandit is best.

load('res_group_s2.mat');

% Compensate for possible partial result sets
pull_counts1 = pull_counts1(1:t_num,:);
pull_counts2 = pull_counts2(1:t_num,:);
pull_counts3 = pull_counts3(1:t_num,:);
all_conf_times = all_conf_times(1:t_num,:,:);
a1_conf_times = all_conf_times(1:t_num,1:a1_count,:);

% Plot CDF figure for completion times for bandits with first arm best
method_styles = {'-','-','-'};
method_colors = [0 0 0; 0.4 0.4 0.4; 0.8 0.8 0.8];
x_max = trial_rounds;
y_max = max(sum(squeeze(sum(a1_conf_times < trial_rounds,1))));

figure();
axes('FontSize', 14);
axis([1 x_max 0 y_max]);
hold on;
plot([0 1e-2], [0 1e-2], method_styles{1}, 'Color', method_colors(1,:), ...
    'LineWidth', 2);
plot([0 1e-2], [0 1e-2], method_styles{2}, 'Color', method_colors(2,:), ...
    'LineWidth', 2);
plot([0 1e-2], [0 1e-2], method_styles{3}, 'Color', method_colors(3,:), ...
    'LineWidth', 2);
legend('Greedy-Bayes','MAP-UCB','Uniform-UCB', 'Location', 'Best');
for i=1:3,
    T = 1:100:trial_rounds;
    if (T(end) < trial_rounds)
        T(end+1) = trial_rounds;
    end
    conf_times = squeeze(a1_conf_times(:,:,i));
    conf_times = conf_times(:);
    style = method_styles{i};
    color = method_colors(i,:);
    y = arrayfun(@( t ) sum(conf_times < t), T);
    line('XData', T, 'YData', y, 'Color', color, ...
        'LineWidth', 2, 'LineStyle', style);
end
xlabel('Trials Allocated','fontsize',14);
ylabel('Confident Results Achieved','fontsize',14);
title('Scenario 2: First Arm Results','fontsize',14);

% Plot CDF figure for completion times for bandits with any arm best
method_styles = {'-','-','-'};
method_colors = [0 0 0; 0.4 0.4 0.4; 0.8 0.8 0.8];
x_max = trial_rounds;
y_max = max(sum(squeeze(sum(all_conf_times < trial_rounds,1))));

figure();
axes('FontSize', 14);
axis([1 x_max 0 y_max]);
hold on;
plot([0 1e-2], [0 1e-2], method_styles{1}, 'Color', method_colors(1,:), ...
    'LineWidth', 2);
plot([0 1e-2], [0 1e-2], method_styles{2}, 'Color', method_colors(2,:), ...
    'LineWidth', 2);
plot([0 1e-2], [0 1e-2], method_styles{3}, 'Color', method_colors(3,:), ...
    'LineWidth', 2);
legend('Greedy-Bayes','MAP-UCB','Uniform-UCB', 'Location', 'Best');
for i=1:3,
    T = 1:100:trial_rounds;
    if (T(end) < trial_rounds)
        T(end+1) = trial_rounds;
    end
    conf_times = squeeze(all_conf_times(:,:,i));
    conf_times = conf_times(:);
    style = method_styles{i};
    color = method_colors(i,:);
    y = arrayfun(@( t ) sum(conf_times < t), T);
    line('XData', T, 'YData', y, 'Color', color, ...
        'LineWidth', 2, 'LineStyle', style);
end
xlabel('Trials Allocated','fontsize',14);
ylabel('Confident Results Achieved','fontsize',14);
title('Scenario 2: Any Arm Results','fontsize',14);