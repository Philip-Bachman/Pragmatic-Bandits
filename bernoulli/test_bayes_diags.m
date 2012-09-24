% clear; close all;
% 
% % Bandit structure parameters
% group_count = 1;
% arm_count = 10;
% init_rounds = 10 * (group_count * arm_count);
% a_0 = 1;
% b_0 = 1;
% trial_rounds = 8000;
% pred_rounds = 50:25:5000;
% top_ms = [1 3 5];
% test_count = 250;
% val_samples = 50;
% 
% for m=1:numel(top_ms),
%     top_m = top_ms(m);
%     % Result recording arrays
%     comp_costs = zeros(test_count,1);
%     gap_sizes = zeros(test_count,1); 
%     gap_costs = zeros(test_count, numel(pred_rounds), val_samples);
%     sum_costs = zeros(test_count, numel(pred_rounds), val_samples);
%     opt_costs = zeros(test_count, numel(pred_rounds), val_samples);
%     for t_num=1:test_count,
%         fprintf('==================================================\n');
%         fprintf('Test %d, top_m: %d\n',t_num, top_m);
%         fprintf('==================================================\n');
%         % Make a bandit maker, a bandit, and an optimizer
%         bandit_maker = BernoulliMaker(group_count, arm_count, top_m);
%         bandit = bandit_maker.distro_5(2^(-5), 2^(-2));
%         opt = BayesDiagsOpt(bandit, top_m, a_0, b_0);
%         % Run the optimizer on the bandit
%         [ res ] = opt.run_trials(trial_rounds,init_rounds,0.0,1.0,0.0);
%         % Process results for this test
%         gap_sizes(t_num) = bandit.arm_groups(1,top_m).return - ...
%             bandit.arm_groups(1,top_m+1).return;
%         sprobs = conv(res.group_sprobs, ones(1,300)./300, 'same');
%         if (max(sprobs) > 0.95)
%             comp_costs(t_num) = find(sprobs > 0.95,1,'first');
%         else
%             comp_costs(t_num) = trial_rounds;
%         end
%         gap_costs(t_num,:,:) = -res.group_gaps(pred_rounds,1:val_samples);
%         sum_costs(t_num,:,:) = -res.group_sums(pred_rounds,1:val_samples);
%         opt_costs(t_num,:,:) = res.opt_costs(pred_rounds,1:val_samples);
%         fprintf('GAP: %.4f, COST: %d\n',gap_sizes(t_num),comp_costs(t_num));
%     end
%     fname = sprintf('res_bayes_diags_10x%d.mat',top_m);
%     save(fname);
% end
    
hits = zeros(1,2500);
val_samples = size(gap_costs,3);
for j=1:2:numel(pred_rounds),
    t = pred_rounds(j);
    valid_idx = find(comp_costs > t);
    y_sum = 0;
    for i=1:numel(hits),
        x = randsample(valid_idx,1);
        valid_y = valid_idx(abs(comp_costs(valid_idx) - comp_costs(x)) > 500);
        y_sum = y_sum + numel(valid_y);
        y = randsample(valid_y,1);
        idx_x = randsample(1:val_samples, 20);
        idx_y = randsample(1:val_samples, 20);
        %x_val = mean(squeeze(gap_costs(x,j,idx_x)));
        %y_val = mean(squeeze(gap_costs(y,j,idx_y)));
        x_val = gap_costs(x,j,randi(val_samples));
        y_val = gap_costs(y,j,randi(val_samples));
        if (x_val >= y_val)
            hits(i) = (comp_costs(x) >= comp_costs(y));
        else
            hits(i) = (comp_costs(x) < comp_costs(y));
        end
    end
    fprintf('round: %d, tests: %d, hit_rate: %.4f, barf: %.4f\n',...
        t, numel(valid_idx), sum(hits)/numel(hits), y_sum/numel(hits));
end
