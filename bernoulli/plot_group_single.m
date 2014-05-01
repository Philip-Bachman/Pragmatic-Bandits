% Do plotting for smoothly gapped greedy multiple bandit selection tests

load('res_group_single.mat');

% Compensate for possible partial result sets
pull_counts1 = pull_counts1(1:t_num,:);
pull_counts2 = pull_counts2(1:t_num,:);
pull_counts3 = pull_counts3(1:t_num,:);
all_sprobs1 = all_sprobs1(1:t_num,:);
all_sprobs2 = all_sprobs2(1:t_num,:);
all_sprobs3 = all_sprobs3(1:t_num,:);
all_sprobs = zeros(size(all_sprobs1,1),size(all_sprobs1,2),3);
all_sprobs(:,:,1) = all_sprobs1;
all_sprobs(:,:,2) = all_sprobs2;
all_sprobs(:,:,3) = all_sprobs3;

% Compute group completion times for all three methods, for all valid tests
all_conf_times = zeros(t_num, 3);
for algo=1:3,
    for t=1:t_num,
        g_conf = squeeze(all_sprobs(t,:,algo));
        if (max(g_conf) > 0.98)
            all_conf_times(t,algo) = find(g_conf > 0.98,1,'first');
        else
            all_conf_times(t,algo) = trial_rounds;
        end
    end
end  

% Plot CDF figure for per-algorithm completion times
method_styles = {'-','-','-'};
method_colors = [0 0 0; 0.4 0.4 0.4; 0.7 0.7 0.7];

figure();
axes('FontSize', 18);
axis([1 trial_rounds 0 t_num]);
hold on;
plot([0 1e-2], [0 1e-2], method_styles{1}, 'Color', method_colors(1,:), ...
    'LineWidth', 4);
plot([0 1e-2], [0 1e-2], method_styles{2}, 'Color', method_colors(2,:), ...
    'LineWidth', 4);
plot([0 1e-2], [0 1e-2], method_styles{3}, 'Color', method_colors(3,:), ...
    'LineWidth', 4);
legend('GCP-Bayes','Uni-UCB','Gab-UCB', 'Location', 'NorthWest');
for i=1:3,
    T = 1:100:trial_rounds;
    style = method_styles{i};
    color = method_colors(i,:);
    y = arrayfun(@( t ) sum(all_conf_times(:,i) <= t), T);
    line('XData', T, 'YData', y, 'Color', color, ...
        'LineWidth', 4, 'LineStyle', style);
end
xlabel('Trials Allocated','fontsize',18);
ylabel('Bandits Completed','fontsize',18);
title('Group Selection (one easy)','fontsize',18);
    
    