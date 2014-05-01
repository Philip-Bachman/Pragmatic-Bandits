clear; close all;

arm_counts = [5 10 20 50 75 100];
test_count = 1;
group_count = 1;
top_m = 1;
a_0 = 1;
b_0 = 1;

for ac_num=1:numel(arm_counts),
    % Initialize arm count dependent parameters
    arm_count = arm_counts(ac_num);
    if (arm_count < 50)
        delay = 100;
    else
        delay = 200;
    end
    init_rounds = 10 * (group_count * arm_count);
    trial_rounds = 15000  + (round(arm_count/2) * 1000);
    bandit_maker = BernoulliMaker(group_count, arm_count, top_m);
    % Initialize arrays to hold test results
    group_confs1 = zeros(test_count, trial_rounds);
    group_confs2 = zeros(test_count, trial_rounds);
    group_confs3 = zeros(test_count, trial_rounds);
    pull_counts1 = zeros(test_count, arm_count);
    pull_counts2 = zeros(test_count, arm_count);
    pull_counts3 = zeros(test_count, arm_count);
    succ_probs1 = zeros(test_count, trial_rounds);
    succ_probs2 = zeros(test_count, trial_rounds);
    succ_probs3 = zeros(test_count, trial_rounds);
    comp_times1 = ones(test_count, 1) .* trial_rounds;
    comp_times2 = ones(test_count, 1) .* trial_rounds;
    comp_times3 = ones(test_count, 1) .* trial_rounds;
    for t_num=1:test_count,
        fprintf('==================================================\n');
        fprintf('TEST ROUND %d\n',t_num);
        fprintf('==================================================\n');
        bandit = bandit_maker.distro_5(0.05, 0.15);
        fprintf('BayesTopMOpt Trials:\n');
        opt1 = BayesTopMOpt(bandit, top_m, a_0, b_0);
        opt1.exp_rate = 0.0;
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
        % Check the completion times for each bandit
        if (max(succ_probs1(t_num,:)) > 0.98)
            for r=101:trial_rounds,
                if (min(succ_probs1(t_num,(r-100:r))) > 0.98)
                    comp_times1(t_num) = r;
                    break
                end
            end
        end
        if (max(succ_probs2(t_num,:)) > 0.98)
            for r=101:trial_rounds,
                if (min(succ_probs2(t_num,(r-100:r))) > 0.98)
                    comp_times2(t_num) = r;
                    break
                end
            end
        end
        if (max(succ_probs3(t_num,:)) > 0.98)
            for r=101:trial_rounds,
                if (min(succ_probs3(t_num,(r-100:r))) > 0.98)
                    comp_times3(t_num) = r;
                    break
                end
            end
        end
    end
    fname = sprintf('res_arm_counts_delay_%d.mat',arm_count);
    save(fname);
end

