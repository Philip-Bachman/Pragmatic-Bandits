clear; close all;

test_count = 1;
trial_rounds = 2000;
group_count = 12;
arm_count = 6;
top_m = 3;
init_rounds = 5 * (group_count * arm_count);
a_0 = 1;
b_0 = 1;
k_0 = 1;
bandit_maker = NormalMaker(group_count, arm_count);

sig_counts0 = zeros(test_count,trial_rounds);
sig_counts1 = zeros(test_count,trial_rounds);
% fig = figure();
for t_num=1:test_count
    fprintf('==================================================\n');
    fprintf('TEST ROUND %d\n',t_num);
    fprintf('==================================================\n');
    %
    bandit = bandit_maker.distro_2(top_m, 0.1, 0.1, 0.0);
    %
    opt0 = TopMNOpt(bandit, top_m, a_0, b_0, k_0);
    [sc0 gc0] = opt0.run_trials(trial_rounds,init_rounds,0.2,0.99,0.0);
    sig_counts0(t_num,:) = sc0(:);
    opt1 = TopMNOpt(bandit, top_m, a_0, b_0, k_0);
    [sc1 gc1] = opt1.run_trials(trial_rounds,init_rounds,0.2,0.50,0.75);
    sig_counts1(t_num,:) = sc1(:);
end
    