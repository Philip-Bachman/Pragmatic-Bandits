clear; close all;

arm_count = 10;
test_count = 1;
group_count = 1;
delay = 1;
top_m = 1;
a_0 = 1;
b_0 = 1;


init_rounds = 10 * (group_count * arm_count);
trial_rounds = 1000;
bandit_maker = BernoulliMaker(group_count, arm_count, top_m);
% Initialize arrays to hold test results
group_confs1 = zeros(test_count, trial_rounds);
pull_counts1 = zeros(test_count, arm_count);
succ_probs1 = zeros(test_count, trial_rounds);
for t_num=1:test_count,
    fprintf('==================================================\n');
    fprintf('TEST ROUND %d\n',t_num);
    fprintf('==================================================\n');
    bandit = bandit_maker.distro_5(0.05, 0.06);
    fprintf('BayesTopMOpt Trials:\n');
    opt1 = BayesTopMOpt(bandit, top_m, a_0, b_0);
    opt1.delay = delay;
    opt1.exp_rate = 0.0;
    [ res1 ] = opt1.run_slim_trials(trial_rounds,init_rounds,0.0);
    group_confs1(t_num,:) = res1.group_confs(:);
    succ_probs1(t_num,:) = res1.group_sprobs(:);
    % For each optimizer, count the number of times it pulled each arm
    for a=1:arm_count,
        pull_counts1(t_num,a) = numel(opt1.bandit_stats(1,a).pulls);
    end
end

