% Do plotting for subset selection tests

fnames = {'res_subset_10x1.mat','res_subset_10x3.mat','res_subset_10x5.mat',...
    'res_subset_20x1.mat','res_subset_20x5.mat','res_subset_20x10.mat',...
    'res_subset_50x1.mat','res_subset_50x10.mat','res_subset_50x25.mat'};
subset_colors = [0.0 0.6 0.8];

% % Plot figure for 10-armed bandits and various subset sizes
% plot_idx = [1 3]; %[1 2 3];
% figure();
% axes('FontSize', 14);
% axis([1 8000 0 1]);
% hold on;
% plot([0 1e-2], [0 1e-2], 'k-', 'LineWidth', 2);
% plot([0 1e-2], [0 1e-2], 'k--', 'LineWidth', 2);
% plot([0 1e-2], [0 1e-2], 'k-.', 'LineWidth', 2);
% legend('Bayes','UCB','PAC', 'Location', 'Best');
% for i=1:numel(plot_idx),
%     idx = plot_idx(i);
%     load(fnames{idx});
%     keeps = 1:100:size(succ_probs1,2);
%     c = subset_colors(i);
%     line('XData', keeps, 'YData', mean(succ_probs1(:,keeps)),...
%         'Color', [c c c], 'LineWidth', 2, 'LineStyle', '-');
%     line('XData', keeps, 'YData', mean(succ_probs2(:,keeps)),...
%         'Color', [c c c], 'LineWidth', 2, 'LineStyle', '--');
%     line('XData', keeps, 'YData', mean(succ_probs3(:,keeps)),...
%         'Color', [c c c], 'LineWidth', 2, 'LineStyle', '-.');
% end
% xlabel('Trials Allocated','fontsize',14);
% ylabel('Selection Confidence','fontsize',14);
% title('10-Armed Bandit Learning Rates','fontsize',14);

% % Plot figure for 20-armed bandits and various subset sizes
% plot_idx = [4 6];
% figure();
% axes('FontSize', 14);
% axis([1 15000 0 1]);
% hold on;
% plot([0 1e-2], [0 1e-2], 'k-', 'LineWidth', 2);
% plot([0 1e-2], [0 1e-2], 'k--', 'LineWidth', 2);
% plot([0 1e-2], [0 1e-2], 'k-.', 'LineWidth', 2);
% legend('Bayes','UCB','PAC', 'Location', 'Best');
% for i=1:numel(plot_idx),
%     idx = plot_idx(i);
%     load(fnames{idx});
%     keeps = 1:100:size(succ_probs1,2);
%     c = subset_colors(i);
%     line('XData', keeps, 'YData', mean(succ_probs1(:,keeps)),...
%         'Color', [c c c], 'LineWidth', 2, 'LineStyle', '-');
%     line('XData', keeps, 'YData', mean(succ_probs2(:,keeps)),...
%         'Color', [c c c], 'LineWidth', 2, 'LineStyle', '--');
%     line('XData', keeps, 'YData', mean(succ_probs3(:,keeps)),...
%         'Color', [c c c], 'LineWidth', 2, 'LineStyle', '-.');
% end
% xlabel('Trials Allocated','fontsize',14);
% ylabel('Selection Confidence','fontsize',14);
% title('20-Armed Bandit Learning Rates','fontsize',14);

% % Plot figure for 50-armed bandits and various subset sizes
% plot_idx = [7 9];
% figure();
% axes('FontSize', 14);
% axis([1 30000 0 1]);
% hold on;
% plot([0 1e-2], [0 1e-2], 'k-', 'LineWidth', 2);
% plot([0 1e-2], [0 1e-2], 'k--', 'LineWidth', 2);
% plot([0 1e-2], [0 1e-2], 'k-.', 'LineWidth', 2);
% legend('Bayes','UCB','PAC', 'Location', 'Best');
% for i=1:numel(plot_idx),
%     idx = plot_idx(i);
%     load(fnames{idx});
%     keeps = 1:100:size(succ_probs1,2);
%     c = subset_colors(i);
%     line('XData', keeps, 'YData', mean(succ_probs1(:,keeps)),...
%         'Color', [c c c], 'LineWidth', 2, 'LineStyle', '-');
%     line('XData', keeps, 'YData', mean(succ_probs2(:,keeps)),...
%         'Color', [c c c], 'LineWidth', 2, 'LineStyle', '--');
%     line('XData', keeps, 'YData', mean(succ_probs3(:,keeps)),...
%         'Color', [c c c], 'LineWidth', 2, 'LineStyle', '-.');
% end
% xlabel('Trials Allocated','fontsize',14);
% ylabel('Selection Confidence','fontsize',14);
% title('50-Armed Bandit Learning Rates','fontsize',14);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot some stuff for illustrating the pull distributions of each method
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;
load('res_subset_20x1.mat');
all_pull_counts = zeros(3, arm_count);
all_pull_counts(1,:) = mean(pull_counts1);
all_pull_counts(2,:) = mean(pull_counts2);
all_pull_counts(3,:) = mean(pull_counts3);
all_pull_counts = bsxfun(@rdivide, all_pull_counts, sum(all_pull_counts,2));
pull_colors = [0.0 0.4 0.8];

figure();
axes('FontSize', 14);
axis([1 arm_count 0 max(all_pull_counts(:))]);
hold on;
for i=1:3,
    c = pull_colors(i);
    line('XData', 1:arm_count, 'YData', all_pull_counts(i,:),...
        'Color', [c c c], 'LineWidth', 2, 'LineStyle', '-');
end
legend('Bayes','UCB','PAC', 'Location', 'Best');
xlabel('Arm Number','fontsize',14);
ylabel('Allocation Rate','fontsize',14);
title('Allocation Distributions: Select 1 of 20','fontsize',14);

clear;
load('res_subset_50x10.mat');
all_pull_counts = zeros(3, arm_count);
all_pull_counts(1,:) = mean(pull_counts1);
all_pull_counts(2,:) = mean(pull_counts2);
all_pull_counts(3,:) = mean(pull_counts3);
all_pull_counts = bsxfun(@rdivide, all_pull_counts, sum(all_pull_counts,2));
pull_colors = [0.0 0.4 0.8];

figure();
axes('FontSize', 14);
axis([1 arm_count 0 max(all_pull_counts(:))]);
hold on;
for i=1:3,
    c = pull_colors(i);
    line('XData', 1:arm_count, 'YData', all_pull_counts(i,:),...
        'Color', [c c c], 'LineWidth', 2, 'LineStyle', '-');
end
legend('Bayes','UCB','PAC', 'Location', 'Best');
xlabel('Arm Number','fontsize',14);
ylabel('Allocation Rate','fontsize',14);
title('Allocation Distributions: Select 10 of 50','fontsize',14);
    
    