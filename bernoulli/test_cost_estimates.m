clear; close all;

for top_m=1:5,

    test_count = 100;
    sim_count = 15;
    trial_rounds = 7500;
    group_count = 1;
    arm_count = 10;
    init_rounds = 10 * (group_count * arm_count);
    a_0 = 1;
    b_0 = 1;
    bandit_maker = BernoulliMaker(group_count, arm_count, top_m);

    all_confs1 = zeros(test_count, sim_count, trial_rounds);
    succ_probs1 = zeros(test_count, sim_count, trial_rounds);
    succ_probs0 = zeros(test_count, trial_rounds);
    for t_num=1:test_count,
        % Create a bandit archetype to reproduce for these simulations
        bandit = bandit_maker.distro_5(0.05, 0.15);
        b_returns = zeros(group_count,arm_count);
        for g=1:group_count,
            for a=1:arm_count,
                b_returns(g,a) = bandit.arm_groups(g,a).return;
            end
        end
        % Compute the "optimator" estimate of the archetypal bandit's cost
        [ res_opt ] = StaticTopMOpt.optimate_bandit(...
            bandit, 1, top_m, trial_rounds, init_rounds);
        succ_probs0(t_num,:) = max(0, (1 - res_opt.fail_probs));
        for s_num=1:sim_count,
            fprintf('==================================================\n');
            fprintf('TEST %d, SIMULATION %d/%d:\n',t_num, s_num, sim_count);
            fprintf('==================================================\n');
            bandit = BernoulliMaker.clone_returns(b_returns);
            fprintf('BayesTopMOpt Trials:\n');
            opt1 = BayesTopMOpt(bandit, top_m, a_0, b_0);
            [ res1 ] = opt1.run_slim_trials(trial_rounds, init_rounds, 0.0);
            all_confs1(t_num,s_num,:) = res1.group_confs(:);
            succ_probs1(t_num,s_num,:) = res1.group_sprobs(:);
        end
    end

    % Do some analysis
    filt = ones(1,300) ./ 300;
    all_confs2 = zeros(size(all_confs1));
    succ_probs2 = zeros(size(succ_probs1));
    for i=1:test_count,
        for j=1:sim_count,
            all_confs2(i,j,:)=conv(squeeze(all_confs1(i,j,:)),filt,'same');
            succ_probs2(i,j,:)=conv(squeeze(succ_probs1(i,j,:)),filt,'same');
        end
    end

    % Compute estimated confidence times
    conf_times_opt = zeros(test_count,1);
    conf_times_bayes = zeros(test_count,sim_count);
    conf_times_union = zeros(test_count,sim_count);
    for i=1:test_count,
        % Compute confidence time estimated by optimator
        try
            conf_times_opt(i) = find(succ_probs0(i,:) > 0.95,1,'first');
        catch
            conf_times_opt(i) = trial_rounds;
        end
        for j=1:sim_count,
            % Compute confidence time using bayesian confidence test
            try
                conf_times_bayes(i,j) = find(squeeze(all_confs2(i,j,:)) > 0.99,1,'first');
            catch
                conf_times_bayes(i,j) = trial_rounds;
            end
            % Compute confidence time using union-bound-based test
            try
                conf_times_union(i,j) = find(squeeze(succ_probs2(i,j,:)) > 0.95,1,'first');
            catch
                conf_times_union(i,j) = trial_rounds;
            end
        end
    end
    fname = sprintf('res_cost_estimates_10x%d.mat',top_m);
    save(fname);
    
end