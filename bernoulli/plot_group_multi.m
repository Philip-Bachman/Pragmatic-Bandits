% Do plotting for smoothly gapped greedy multiple bandit selection tests

load('res_group_multi.mat');

% Compensate for possible partial result sets
pull_counts1 = pull_counts1(1:t_num,:);
pull_counts2 = pull_counts2(1:t_num,:);
pull_counts3 = pull_counts3(1:t_num,:);


% % Plot CDF figure for per-algorithm completion times
% method_styles = {'-','-','-'};
% method_colors = [0 0 0; 0.4 0.4 0.4; 0.8 0.8 0.8];
% 
% figure();
% axes('FontSize', 14);
% axis([1 trial_rounds 0 t_num]);
% hold on;
% plot([0 1e-2], [0 1e-2], method_styles{1}, 'Color', method_colors(1,:), ...
%     'LineWidth', 2);
% plot([0 1e-2], [0 1e-2], method_styles{2}, 'Color', method_colors(2,:), ...
%     'LineWidth', 2);
% plot([0 1e-2], [0 1e-2], method_styles{3}, 'Color', method_colors(3,:), ...
%     'LineWidth', 2);
% legend('Greedy-Bayes','Uniform-UCB','Lower-UCB', 'Location', 'Best');
% for i=1:3,
%     T = 1:100:trial_rounds;
%     conf_times = squeeze(all_conf_times(:,:,i));
%     conf_times = conf_times(:);
%     style = method_styles{i};
%     color = method_colors(i,:);
%     y = arrayfun(@( t ) sum(conf_times <= t), T);
%     line('XData', T, 'YData', y, 'Color', color, ...
%         'LineWidth', 2, 'LineStyle', style);
% end
% xlabel('Trials Allocated','fontsize',14);
% ylabel('Confident Results Achieved','fontsize',14);
% title('Comparing Group Selection Methods','fontsize',14);