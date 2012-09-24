clear; close all;

test_count = 5000;
trial_rounds = 10000;
group_count = 1;
arm_count = 10;
init_rounds = 10 * (group_count * arm_count);
a_0 = 1;
b_0 = 1;

%succ_probs = zeros(test_count, trial_rounds);
opti_costs = zeros(test_count, 1);
comp_costs = zeros(test_count, 1);
ngap_costs = zeros(test_count, 1);
bandit_gaps = zeros(test_count, 1);
bandit_topm = zeros(test_count, 1);
for t_num=1:test_count,
    if ((mod(t_num, 100) == 0) || (t_num == 1))
        fprintf('Test %d: ',t_num);
    else
        if (mod(t_num, 2) == 0)
            fprintf('.');
        end
        if (mod(t_num, 100) == 99)
            fprintf('\n');
        end
    end
    % Create a bandit for which to compute a set of complexity estimates
    top_m = round(t_num / 1000) + 1;
    bandit_maker = BernoulliMaker(group_count, arm_count, top_m);
    bandit = bandit_maker.distro_5(2^(-5), 2^(-2));
    % Get basic per-arm stats for this bandit
    arm_returns = zeros(1,arm_count);
    arm_vars = zeros(1,arm_count);
    for a=1:arm_count,
        arm_returns(a) = bandit.arm_groups(1,a).return;
        arm_vars(a) = arm_returns(a) * (1 - arm_returns(a));
    end
    % Record the gap for this bandit
    bandit_gaps(t_num) = arm_returns(top_m) - arm_returns(top_m + 1);
    bandit_topm(t_num) = top_m;
    % Compute the "optimator" complexity estimate for this bandit
    [ res_opt ] = StaticTopMOpt.optimate_bandit(...
        bandit, 1, top_m, trial_rounds, init_rounds);
    succ_probs = max(0, (1 - res_opt.fail_probs));
    if (max(succ_probs) > 0.95)
        opti_costs(t_num) = find(succ_probs > 0.95,1,'first');
    else
        opti_costs(t_num) = trial_rounds;
    end
    % Compute the UCB-ish complexity estimate for this bandit
    arm_comps = zeros(1,arm_count);
    for a=1:arm_count,
        if (a <= top_m)
            a_gap = arm_returns(a) - arm_returns(top_m + 1);
        else
            a_gap = arm_returns(top_m) - arm_returns(a);
        end
        a_var = arm_vars(a);
        arm_comps(a) = (sqrt(a_var) + sqrt(a_var + (16/3)*a_gap))^2 / a_gap^2;
    end
    comp_costs(t_num) = sum(arm_comps);
    % Compute the simple normalized gap complexity estimate for this bandit
    ngap_costs(t_num) = (arm_returns(top_m) + arm_returns(top_m + 1)) / ...
        sqrt(arm_vars(top_m) + arm_vars(top_m + 1));
end