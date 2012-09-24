clear; close all;

test_count = 75;
trial_rounds = 700;
group_count = 2;
arm_count = 4;
top_m = 1;
init_rounds = 10 * (group_count * arm_count);
a_0 = 1;
b_0 = 1;
bandit_maker = BernoulliMaker(group_count, arm_count, top_m);


accs_bayes = zeros(1,test_count);
accs_ucb = zeros(1,test_count);
confs_bayes = zeros(test_count, group_count, trial_rounds);
confs_ucb = zeros(test_count, group_count, trial_rounds);
for t_num=1:test_count
    fprintf('==================================================\n');
    fprintf('TEST ROUND %d\n',t_num);
    fprintf('==================================================\n');
    bandit = bandit_maker.mbbai_problem1();
    fprintf('BAYES TRIALS:\n');
    opt_bayes = UCBTopMOpt(bandit, top_m, a_0, b_0);
    opt_bayes.do_bayes = 1; opt_bayes.exp_rate = 0.0;
    res_bayes = opt_bayes.run_trials(trial_rounds,init_rounds,0.0,1.0,0.0);
    confs_bayes(t_num,:,:) = res_bayes.group_confs;
    accs_bayes(t_num) = res_bayes.select_accs(end);
    fprintf('UCB TRIALS:\n');
    opt_ucb = UCBTopMOpt(bandit, top_m, a_0, b_0);
    opt_ucb.do_bayes = 0;
    res_ucb = opt_ucb.run_trials(trial_rounds,init_rounds,0.0,1.0,0.0);
    confs_ucb(t_num,:,:) = res_ucb.group_confs;
    accs_ucb(t_num) = res_ucb.select_accs(end);
    
end
    