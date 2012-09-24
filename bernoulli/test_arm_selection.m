clear; close all;

test_count = 30;
trial_rounds = 3500;
group_count = 1;
arm_count = 20;
top_m = 5;
init_rounds = 10 * (group_count * arm_count);
a_0 = 1;
b_0 = 1;
bandit_maker = BernoulliMaker(group_count, arm_count, top_m);

all_confs1 = zeros(test_count, trial_rounds);
all_confs2 = zeros(test_count, trial_rounds);
all_confs3 = zeros(test_count, trial_rounds);
pull_counts1 = zeros(test_count, arm_count);
pull_counts2 = zeros(test_count, arm_count);
pull_counts3 = zeros(test_count, arm_count);
for t_num=1:test_count
    fprintf('==================================================\n');
    fprintf('TEST ROUND %d\n',t_num);
    fprintf('==================================================\n');
    bandit = bandit_maker.distro_5(0.06, 0.08);
    fprintf('BAYES TRIALS:\n');
    opt1 = BayesTopMOpt(bandit, top_m, a_0, b_0);
    [ res1 ] = opt1.run_trials(trial_rounds,init_rounds,0.0,0.995,0.5);
    all_confs1(t_num,:) = res1.group_confs(:);
    fprintf('PAC TRIALS:\n');
    opt2 = PACTopMOpt(bandit, top_m, a_0, b_0);
    [ res2 ] = opt2.run_trials(trial_rounds,init_rounds,0.0,0.995,0.5);
    all_confs2(t_num,:) = res2.group_confs(:);
    fprintf('UCB TRIALS:\n');
    opt3 = UCBTopMOpt(bandit, top_m, a_0, b_0);
    [ res3 ] = opt3.run_trials(trial_rounds,init_rounds,0.0,0.995,0.5);
    all_confs3(t_num,:) = res3.group_confs(:);
    % For each optimizer, count the number of times it pulled each arm
    for a=1:arm_count,
        pull_counts1(t_num,a) = numel(opt1.bandit_stats(1,a).pulls);
        pull_counts2(t_num,a) = numel(opt2.bandit_stats(1,a).pulls);
        pull_counts3(t_num,a) = numel(opt3.bandit_stats(1,a).pulls);
    end
end
    