clear; close all;

test_count = 50;
trial_rounds = 1000;
group_count = 1;
arm_count = 15;
init_rounds = 10 * (group_count * arm_count);
top_m = 5;
alpha = 1.0;
beta = 1.0;
kappa = 1.0;
bandit_maker = NormalMaker(group_count, arm_count);

t_confs0 = zeros(test_count,trial_rounds);
t_confs1 = zeros(test_count,trial_rounds);
t_confs2 = zeros(test_count,trial_rounds);
t_pulls0 = zeros(test_count,arm_count);
t_pulls1 = zeros(test_count,arm_count);
fig = figure();
for t_num=1:test_count
    fprintf('==================================================\n');
    fprintf('TEST ROUND %d\n',t_num);
    fprintf('==================================================\n');
    %bandit = bandit_maker.distro_1();
    %bandit = bandit_maker.distro_2(top_m, 0.2, 0.1, 0.75);
    bandit = bandit_maker.distro_3(top_m, 0.15, 0.45);
    opt2 = TopMNOpt(bandit, top_m, alpha, beta, kappa);
    [sig_counts group_confs2] = ...
        opt2.run_uniform_trials(trial_rounds, 1.0);
    t_confs2(t_num,:) = group_confs2(:);
    opt1 = UniMNOpt(bandit, top_m, alpha, beta, kappa);
    [sig_counts group_confs1] = ...
        opt1.run_trials(trial_rounds,init_rounds,0.2,0.6,0.75);
    t_confs1(t_num,:) = group_confs1(:);
    opt0 = TopMNOpt(bandit, top_m, alpha, beta, kappa);
    [sig_counts group_confs0] = ...
        opt0.run_trials(trial_rounds,init_rounds,0.2,1.0,0.9);
    t_confs0(t_num,:) = group_confs0(:);
    figure(fig); clf; cla; hold on;
    if (t_num == 1)
        plot(t_confs0(1,:),'b-');
        plot(t_confs1(1,:),'g-');
        plot(t_confs2(1,:),'r-');
    else
        plot(mean(t_confs0(1:t_num,:)),'b-');
        plot(mean(t_confs1(1:t_num,:)),'g-');
        plot(mean(t_confs2(1:t_num,:)),'r-');
    end
    drawnow;
    % Count pulls on each arm for opt0 and opt1
    fprintf('ARM PULLS: ');
    for a=1:arm_count,
        t_pulls0(t_num,a) = numel(opt0.bandit_stats(a).pulls);
        t_pulls1(t_num,a) = numel(opt1.bandit_stats(a).pulls);
        fprintf('(%d, %d, %d) ',a,t_pulls0(t_num,a),t_pulls1(t_num,a));
    end
    fprintf('\n');
end





%%%%%%%%%%%%%%
% EYE BUFFER %
%%%%%%%%%%%%%%
    