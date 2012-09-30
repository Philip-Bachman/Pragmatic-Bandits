clear; close all;

test_count = 100;
trial_rounds = 75000;
group_count = 15;
arm_count = 10;
top_m = 1;
init_rounds = 20 * (group_count * arm_count);
a_0 = 1;
b_0 = 1;
bandit_maker = BernoulliMaker(group_count, arm_count, top_m);

all_sprobs1 = zeros(test_count, trial_rounds);
all_sprobs2 = zeros(test_count, trial_rounds);
all_sprobs3 = zeros(test_count, trial_rounds);
pull_counts1 = zeros(test_count, group_count);
pull_counts2 = zeros(test_count, group_count);
pull_counts3 = zeros(test_count, group_count);
for t_num=1:test_count
    tic();
    fprintf('==================================================\n');
    fprintf('TEST ROUND %d\n',t_num);
    fprintf('==================================================\n');
    bandit = bandit_maker.distro_6( logspace(-2,-1,group_count) );
    fprintf('Bayes-Bayes Trials:\n');
    opt1 = BayesTopMOpt(bandit, top_m, a_0, b_0);
    opt1.sig_thresh = 0.985; opt1.gap_samples = 50;
    res1 = opt1.run_slim_trials(trial_rounds,init_rounds,0.5);
    all_sprobs1(t_num,:) = res1.group_sprobs(1,:);
    pull_counts1(t_num,:) = transpose(sum(opt1.get_pull_counts(),2));
    fprintf('MAP-UCB Trials:\n');
    opt2 = UniTopMOpt(bandit, top_m, a_0, b_0);
    opt2.sig_thresh = 0.985;
    res2 = opt2.run_slim_trials(trial_rounds,init_rounds,0.5);
    all_sprobs2(t_num,:) = res2.group_sprobs(1,:);
    pull_counts2(t_num,:) = transpose(sum(opt2.get_pull_counts(),2));
    fprintf('Uniform-UCB Trials:\n');
    opt3 = UniTopMOpt(bandit, top_m, a_0, b_0);
    res3 = opt3.run_slim_trials(trial_rounds,init_rounds,1.0);
    all_sprobs3(t_num,:) = res3.group_sprobs(1,:);
    pull_counts3(t_num,:) = transpose(sum(opt3.get_pull_counts(),2));
    save('res_group_multi.mat');
    t=toc(); fprintf('Elapsed time: %.4f\n',t);
end

