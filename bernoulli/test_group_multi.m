clear; close all;

test_count = 100;
trial_rounds = 80000;
group_count = 15;
arm_count = 10;
top_m = 1;
init_rounds = 20 * (group_count * arm_count);
a_0 = 1;
b_0 = 1;
bandit_maker = BernoulliMaker(group_count, arm_count, top_m);

all_conf_times = zeros(test_count, group_count, 3);
pull_counts1 = zeros(test_count, group_count);
pull_counts2 = zeros(test_count, group_count);
pull_counts3 = zeros(test_count, group_count);
for t_num=1:test_count,
    tic();
    fprintf('==================================================\n');
    fprintf('TEST ROUND %d\n',t_num);
    fprintf('==================================================\n');
    bandit = bandit_maker.distro_6( logspace(-2,-1,group_count) );
    fprintf('Bayes-Bayes Trials:\n');
    opt1 = BayesTopMOpt(bandit, top_m, a_0, b_0);
    opt1.sig_thresh = 0.98; opt1.gap_samples = 10;
    res1 = opt1.run_slim_trials(trial_rounds,init_rounds,0.5);
    pull_counts1(t_num,:) = transpose(sum(opt1.get_pull_counts(),2));
    fprintf('MAP-UCB Trials:\n');
    opt2 = UniTopMOpt(bandit, top_m, a_0, b_0);
    opt2.sig_thresh = 0.98;
    res2 = opt2.run_slim_trials(trial_rounds,init_rounds,0.5);
    pull_counts2(t_num,:) = transpose(sum(opt2.get_pull_counts(),2));
    fprintf('Uniform-UCB Trials:\n');
    opt3 = UniTopMOpt(bandit, top_m, a_0, b_0);
    res3 = opt3.run_slim_trials(trial_rounds,init_rounds,1.0);
    pull_counts3(t_num,:) = transpose(sum(opt3.get_pull_counts(),2));
    % Compute completion times for each group for each method
    for i=1:3,
        if (i == 1)
            sp = res1.group_sprobs;
        else
            if (i == 2)
                sp = res2.group_sprobs;
            else
                sp = res3.group_sprobs;
            end
        end
        for g=1:group_count,
            if (max(sp(g,:)) > 0.98)
                all_conf_times(t_num,g,i) = find(sp(g,:) > 0.98, 1, 'first');
            else
                all_conf_times(t_num,g,i) = trial_rounds;
            end
        end
    end
    save('res_group_multi.mat');
    t=toc(); fprintf('Elapsed time: %.4f\n',t);
end

