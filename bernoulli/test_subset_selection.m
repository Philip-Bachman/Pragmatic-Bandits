clear; close all;

param_sets = zeros(9,3);
param_sets(1,:) = [10 1 10000];
param_sets(2,:) = [10 3 10000];
param_sets(3,:) = [10 5 10000];
param_sets(4,:) = [20 1 15000];
param_sets(5,:) = [20 5 15000];
param_sets(6,:) = [20 10 15000];
param_sets(7,:) = [50 1 30000];
param_sets(8,:) = [50 10 30000];
param_sets(9,:) = [50 25 30000];
test_count = 100;
group_count = 1;
a_0 = 1;
b_0 = 1;

for param_set=1:size(param_sets,1),
    % Initialize arm count / subset size dependent parameters
    arm_count = param_sets(param_set,1);
    top_m = param_sets(param_set,2);
    trial_rounds = param_sets(param_set,3);
    init_rounds = 10 * (group_count * arm_count);
    bandit_maker = BernoulliMaker(group_count, arm_count, top_m);
    % Create arrays to hold test results
    group_confs1 = zeros(test_count, trial_rounds);
    group_confs2 = zeros(test_count, trial_rounds);
    group_confs3 = zeros(test_count, trial_rounds);
    pull_counts1 = zeros(test_count, arm_count);
    pull_counts2 = zeros(test_count, arm_count);
    pull_counts3 = zeros(test_count, arm_count);
    succ_probs1 = zeros(test_count, trial_rounds);
    succ_probs2 = zeros(test_count, trial_rounds);
    succ_probs3 = zeros(test_count, trial_rounds);
    for t_num=1:test_count,
        fprintf('==================================================\n');
        fprintf('Test %d (selecting %d/%d arms)\n',t_num,top_m,arm_count);
        fprintf('==================================================\n');
        bandit = bandit_maker.distro_5(0.05, 0.15);
        fprintf('BayesTopMOpt Trials:\n');
        opt1 = BayesTopMOpt(bandit, top_m, a_0, b_0);
        [ res1 ] = opt1.run_slim_trials(trial_rounds,init_rounds,0.0);
        group_confs1(t_num,:) = res1.group_confs(:);
        succ_probs1(t_num,:) = res1.group_sprobs(:);
        fprintf('UCBTopMOpt Trials:\n');
        opt2 = UCBTopMOpt(bandit, top_m, a_0, b_0);
        [ res2 ] = opt2.run_slim_trials(trial_rounds,init_rounds,0.0);
        group_confs2(t_num,:) = res2.group_confs(:);
        succ_probs2(t_num,:) = res2.group_sprobs(:);
        fprintf('PACTopMOpt Trials:\n');
        opt3 = PACTopMOpt(bandit, top_m, a_0, b_0);
        [ res3 ] = opt3.run_slim_trials(trial_rounds,init_rounds,0.0);
        group_confs3(t_num,:) = res3.group_confs(:);
        succ_probs3(t_num,:) = res3.group_sprobs(:);
        % For each optimizer, count the number of times it pulled each arm
        for a=1:arm_count,
            pull_counts1(t_num,a) = numel(opt1.bandit_stats(1,a).pulls);
            pull_counts2(t_num,a) = numel(opt2.bandit_stats(1,a).pulls);
            pull_counts3(t_num,a) = numel(opt3.bandit_stats(1,a).pulls);
        end
    end
    fname = sprintf('res_subset_%dx%d.mat',arm_count,top_m);
    save(fname);
end

