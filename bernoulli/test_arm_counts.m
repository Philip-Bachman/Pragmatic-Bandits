clear; close all;

arm_counts = [5 10 20 50 100];
test_count = 50;
group_count = 1;
top_m = 1;
a_0 = 1;
b_0 = 1;

for ac_num=1:numel(arm_counts),
    % Initialize arm count dependent parameters
    arm_count = arm_counts(ac_num);
    init_rounds = 10 * (group_count * arm_count);
    trial_rounds = 10000  + (round(arm_count/2) * 1000);
    bandit_maker = BernoulliMaker(group_count, arm_count, top_m);
    % Initialize arrays to hold test results
    group_confs1 = zeros(test_count, trial_rounds);
    group_confs2 = zeros(test_count, trial_rounds);
    pull_counts1 = zeros(test_count, arm_count);
    pull_counts2 = zeros(test_count, arm_count);
    succ_probs1 = zeros(test_count, trial_rounds);
    succ_probs2 = zeros(test_count, trial_rounds);
    comp_times1 = ones(test_count, 1) .* trial_rounds;
    comp_times2 = ones(test_count, 1) .* trial_rounds;
    for t_num=1:test_count,
        fprintf('==================================================\n');
        fprintf('TEST ROUND %d\n',t_num);
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
        % For each optimizer, count the number of times it pulled each arm
        for a=1:arm_count,
            pull_counts1(t_num,a) = numel(opt1.bandit_stats(1,a).pulls);
            pull_counts2(t_num,a) = numel(opt2.bandit_stats(1,a).pulls);
        end
        % Check the completion times for each bandit
        if (max(succ_probs1(t_num,:)) > 0.98)
            try
                comp_times1(t_num) = find(succ_probs1(t_num,:) > 0.98,1,'first');
            catch
                fprintf('oops\n');
            end
        end
        if (max(succ_probs2(t_num,:)) > 0.98)
            try
                comp_times2(t_num) = find(succ_probs2(t_num,:) > 0.98,1,'first');
            catch
                fprintf('oops\n');
            end
        end
    end
    fname = sprintf('res_arm_counts_%d.mat',arm_count);
    save(fname);
end

