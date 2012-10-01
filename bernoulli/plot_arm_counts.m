% Plot arm count test results
fnames = {'res_arm_counts_5.mat','res_arm_counts_10.mat',...
    'res_arm_counts_20.mat','res_arm_counts_50.mat','res_arm_counts_100.mat'};
arm_counts = [5 10 20 50 100];
count_colors = [0.1 0.2 0.4 0.6 0.8];

% Plot figure showing full confidence curves for both methods
figure();
axes('FontSize',14);
axis([1 25000 0 1]);
hold on;
plot([0 1e-2], [0 1e-2], 'k-', 'LineWidth', 2);
plot([0 1e-2], [0 1e-2], 'k-.', 'LineWidth', 2);
legend('Bayes','UCB', 'Location', 'Best');
for i=1:numel(fnames),
    load(fnames{i});
    arm_count = arm_counts(i);
    c = count_colors(i);
    line('XData',1:size(succ_probs1,2),'YData',mean(succ_probs1),...
        'Color',[c c c],'LineWidth',2,'LineStyle','-');
    line('XData',1:size(succ_probs1,2),'YData',mean(succ_probs2),...
        'Color',[c c c],'LineWidth',2,'LineStyle','-.');
end
xlabel('Trials Allocated','fontsize',14);
ylabel('Selection Confidence','fontsize',14);
title('Arm Selection: Bayes vs. UCB','fontsize',14);

% Plot figure showing curves for completion time as a function of arm count
comp_stats1 = zeros(numel(fnames),3);
comp_stats2 = zeros(numel(fnames),3);
all_ctimes1 = [];
all_ctimes2 = [];
for i=1:numel(fnames),
    load(fnames{i});
    comp_stats1(i,:) = quantile(comp_times1,[0.2 0.5 0.8]);
    comp_stats2(i,:) = quantile(comp_times2,[0.2 0.5 0.8]);
    all_ctimes1 = [all_ctimes1; comp_times1];
    all_ctimes2 = [all_ctimes2; comp_times2];
end
figure();
axes('FontSize',14);
axis([min(arm_counts) max(arm_counts) 0 max(comp_stats2(:))]);
hold on;
b_col = [0 0 0];
u_col = [0.5 0.5 0.5];
plot([0 1e-2], [0 1e-2], 'Color', b_col, 'LineStyle', '-', 'LineWidth', 2);
plot([0 1e-2], [0 1e-2], 'Color', u_col, 'LineStyle', '-', 'LineWidth', 2);
legend('Bayes','UCB','Location','Best');
% Plot medians
line('XData',arm_counts,'YData',comp_stats1(:,2),'Color',b_col,...
    'LineWidth',2,'LineStyle','-');
line('XData',arm_counts,'YData',comp_stats2(:,2),'Color',u_col,...
    'LineWidth',2,'LineStyle','-');
% Plot upper and lower quintiles
line('XData',arm_counts,'YData',comp_stats1(:,1),'Color',b_col,...
    'LineWidth',2,'LineStyle','--');
line('XData',arm_counts,'YData',comp_stats2(:,1),'Color',u_col,...
    'LineWidth',2,'LineStyle','--');
line('XData',arm_counts,'YData',comp_stats1(:,3),'Color',b_col,...
    'LineWidth',2,'LineStyle','--');
line('XData',arm_counts,'YData',comp_stats2(:,3),'Color',u_col,...
    'LineWidth',2,'LineStyle','--');
% Setup labels
xlabel('Arm Count','fontsize',14);
ylabel('Completion Time','fontsize',14);
title('Best Arm Scaling: Bayes vs. UCB');

    
    