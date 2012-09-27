clear; close all;

test_count = 50;
trial_rounds = 15000;
group_count = 5;
arm_count = 10;
top_m = 1;
init_rounds = 25 * (group_count * arm_count);
a_0 = 1;
b_0 = 1;
bandit_maker = BernoulliMaker(group_count, arm_count, top_m);

%all_confs1 = zeros(test_count, group_count, trial_rounds);
%all_confs2 = zeros(test_count, group_count, trial_rounds);
all_sprobs1 = zeros(test_count, trial_rounds);
all_sprobs2 = zeros(test_count, trial_rounds);
pull_counts1 = zeros(test_count, group_count);
pull_counts2 = zeros(test_count, group_count);
for t_num=1:test_count
    fprintf('==================================================\n');
    fprintf('TEST ROUND %d\n',t_num);
    fprintf('==================================================\n');
    bandit = bandit_maker.distro_2( 0.10, 0.01, (4/5));
    fprintf('BayesTopMOpt Trials:\n');
    opt1 = BayesTopMOpt(bandit, top_m, a_0, b_0);
    opt1.sig_thresh = 0.99; opt1.gap_samples = 50;
    res1 = opt1.run_slim_trials(trial_rounds,init_rounds,0.5);
    %all_confs1(t_num,:,:) = res1.group_confs;
    all_sprobs1(t_num,:) = res1.group_sprobs(1,:);
    pull_counts1(t_num,:) = transpose(sum(opt1.get_pull_counts(),2));
    fprintf('UniTopMOpt Trials:\n');
    opt2 = UniTopMOpt(bandit, top_m, a_0, b_0);
    res2 = opt2.run_slim_trials(trial_rounds,init_rounds,0.0);
    %all_confs2(t_num,:,:) = res2.group_confs;
    all_sprobs2(t_num,:) = res2.group_sprobs(1,:);
    pull_counts2(t_num,:) = transpose(sum(opt2.get_pull_counts(),2));
end
%save('res_bernoulli.mat');

