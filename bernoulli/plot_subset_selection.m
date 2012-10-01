% Do plotting for subset selection tests

fnames = {'res_subset_10x1.mat','res_subset_10x3.mat','res_subset_10x5.mat',...
    'res_subset_20x1.mat','res_subset_20x5.mat','res_subset_20x10.mat',...
    };
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

% Plot figure for 20-armed bandits and various subset sizes
plot_idx = [4 6];
figure();
axes('FontSize', 14);
axis([1 15000 0 1]);
hold on;
plot([0 1e-2], [0 1e-2], 'k-', 'LineWidth', 2);
plot([0 1e-2], [0 1e-2], 'k--', 'LineWidth', 2);
plot([0 1e-2], [0 1e-2], 'k-.', 'LineWidth', 2);
legend('Bayes','UCB','PAC', 'Location', 'Best');
for i=1:numel(plot_idx),
    idx = plot_idx(i);
    load(fnames{idx});
    keeps = 1:100:size(succ_probs1,2);
    c = subset_colors(i);
    line('XData', keeps, 'YData', mean(succ_probs1(:,keeps)),...
        'Color', [c c c], 'LineWidth', 2, 'LineStyle', '-');
    line('XData', keeps, 'YData', mean(succ_probs2(:,keeps)),...
        'Color', [c c c], 'LineWidth', 2, 'LineStyle', '--');
    line('XData', keeps, 'YData', mean(succ_probs3(:,keeps)),...
        'Color', [c c c], 'LineWidth', 2, 'LineStyle', '-.');
end
xlabel('Trials Allocated','fontsize',14);
ylabel('Selection Confidence','fontsize',14);
title('20-Armed Bandit Learning Rates','fontsize',14);
    
    