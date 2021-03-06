classdef BayesDiagsOpt < handle
    % BayesDiagsOpt works a MultiArmBandit of type 'bernoulli'
    % This class performs 'most successful tests' optimization of a
    % MultiArmBandit instance of type 'bernoulli'. The objective, for a given
    % problem instance, is to maximize the number of 'statistically significant
    % results' acheived. For each bandit, the objective is to accurately select
    % the top M arms.
    %
    
    properties
        % bandit is a handle for a MultiArmBandit instance.
        bandit
        % group_count is the number of groups in the current bandit
        group_count
        % arm_count is the number of arms in each group
        arm_count
        % bandit_stats stores per-trial info about arms/groups in self.bandit
        bandit_stats
        % alpha is alpha for initial beta prior over arm returns
        alpha
        % beta is beta for initial beta prior over arm return
        beta
        % sig_thresh is the threshold at which a group's result is significant
        sig_thresh
        % top_m is the number of top arms to try and select
        top_m
        % exp_rate is the rate at which to select arms using posterior variance
        exp_rate
    end
    
    methods
        function [self] = BayesDiagsOpt(ma_bandit, top_m, a_0, b_0)
            % Constructor for multiarmed bandit BayesDiagsOpt 
            if ~exist('a_0','var')
                a_0 = 2.0;
            end
            if ~exist('b_0','var')
                b_0 = 2.0;
            end
            self.alpha = a_0;
            self.beta = b_0;
            self.sig_thresh = 0.99;
            self.set_bandit(ma_bandit);
            self.top_m = top_m;
            self.exp_rate = 0.05;
            return
        end
        
        function [self] = set_bandit(self, ma_bandit)
            % Set the bandit instance to optimize.
            self.bandit = ma_bandit;
            self.bandit.reset_arms(0);
            % initialize optimization structures
            self.group_count = ma_bandit.group_count;
            self.arm_count = ma_bandit.arm_count;
            self.bandit_stats = struct();
            for g=1:self.group_count,
                for a=1:self.arm_count,
                    self.bandit_stats(g,a).sig_thresh = 0.5;
                    self.bandit_stats(g,a).a_n = self.alpha;
                    self.bandit_stats(g,a).b_n = self.beta;
                    pulls = [];
                    for p=1:self.alpha,
                        pulls = [pulls 1];
                    end
                    for p=1:self.beta,
                        pulls = [pulls 0];
                    end
                    self.bandit_stats(g,a).pulls = pulls;
                end
            end
            return
        end
        
        function [ret] = pull_arm(self, group, arm)
            % Pull the given arm in the managed bandit and record stats
            ret = self.bandit.pull_arm(group,arm);
            self.bandit_stats(group,arm).pulls = ...
                [self.bandit_stats(group,arm).pulls ret];
            if (ret == 0)
                self.bandit_stats(group,arm).b_n = ...
                    self.bandit_stats(group,arm).b_n + 1;
            else
                self.bandit_stats(group,arm).a_n = ...
                    self.bandit_stats(group,arm).a_n + 1;
            end
            return
        end
        
        function [pull_idx] = uniform_allocation(self, trial_rounds)
            % Uniformly allocate trial_rounds trials across the current group
            % and arm indices.
            pull_idx = zeros(trial_rounds, 2);
            g = 1;
            a = 1;
            for i=1:trial_rounds,
                pull_idx(i,1) = g;
                pull_idx(i,2) = a;
                [g a] = self.uniform_step(g, a);
            end
            return
        end
        
        function [g_new a_new] = uniform_step(self, g_old, a_old)
            % Take one "step" forward on a "uniform allocation trajectory".
            g_new = g_old;
            a_new = a_old + 1;
            if (a_new > self.arm_count)
                a_new = 1;
                g_new = g_old + 1;
                if (g_new > self.group_count)
                    g_new = 1;
                end
            end
            return
        end
        
        function [best_arms best_returns] = get_best_arms(self, top_m)
            % Get the top_m arms and returns for each group, based on current
            % empirical approximations
            if ~exist('top_m','var')
                top_m = self.top_m;
            end
            best_arms = zeros(self.group_count,top_m);
            best_returns = zeros(self.group_count,top_m);
            for g=1:self.group_count,
                ars = zeros(1,self.arm_count);
                for a=1:self.arm_count,
                    ars(a) = mean(self.bandit_stats(g,a).pulls) + randn()*1e-3;
                end
                [ars ars_idx] = sort(ars,'descend');
                best_arms(g,:) = ars_idx(1:top_m);
                best_returns(g,:) = ars(1:top_m);
            end
            return
        end
        
        function [ arm_mean ] = get_arm_mean(self, group, arm)
            % Get the current empirical mean for the given arm
            as = self.bandit_stats(group,arm);
            if (numel(as.pulls) < 1)
                arm_mean = 0.5;
            else
                arm_mean = mean(as.pulls);
            end
            return
        end
        
        function [ pull_counts ] = get_pull_counts(self)
            % Get the number of pulls peformed for each arm
            pull_counts = zeros(self.group_count, self.arm_count);
            for g=1:self.group_count,
                for a=1:self.arm_count,
                    pull_counts(g,a) = numel(self.bandit_stats(g,a).pulls);
                end
            end
            return
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % GROUP AND ARM PICKING METHODS %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        function [best_group best_gap best_sum] = pick_group(self, group_confs)
            % Pick a group to pull from the managed MultiArmBandit. Ignore the
            % groups in sig_groups (which are already assumed confident).
            sig_groups = find(group_confs > self.sig_thresh);
            arm_mus = zeros(self.group_count,self.arm_count);
            arm_vars = zeros(self.group_count,self.arm_count);
            for g=1:self.group_count,
                for a=1:self.arm_count,
                    as = self.bandit_stats(g,a);
                    r = betarnd(as.a_n, as.b_n);
                    arm_mus(g,a) = r;
                    arm_vars(g,a) = r * (1 - r);
                end
            end
            best_group = randi(self.group_count);
            best_gap = 0;
            for g=1:self.group_count,
                % For group, get the "approximate" gap between arms m and m+1.
                [g_mus g_idx] = sort(arm_mus(g,:),'descend');
                g_vars = arm_vars(g,g_idx);
                g_gap = (g_mus(self.top_m) - g_mus(self.top_m+1)) / ...
                    sqrt(g_vars(self.top_m) + g_vars(self.top_m+1));
                gap_loc = (g_mus(self.top_m) + g_mus(self.top_m+1)) / 2;
                if (g_gap > best_gap)
                    % Only consider groups that are not yet "significant".
                    if ~ismember(g,sig_groups)
                        best_gap = g_gap;
                        best_sum = sum(abs(g_mus - gap_loc) ./ sqrt(g_vars));
                        best_group = g;
                    end
                end
            end
            return
        end
        
        function [best_arm min_gap] = pick_arm(self, group, arm_confs)
            % Pick an arm to pull from the managed MultiArmBandit, given the
            % group from which an arm is to be picked. Pick an arm based on a
            % single sample from the distribution over all arm returns in the
            % group. The arm with the smallest normalized distance to the gap
            % center will be selected.
            a_returns = zeros(1,self.arm_count);
            a_vars = zeros(1,self.arm_count);
            for a=1:self.arm_count,
                as = self.bandit_stats(group,a);
                a_pc = numel(as.pulls);
                a_mu = betarnd(as.a_n, as.b_n, 1);
                a_returns(a) = a_mu;
                a_vars(a) = (a_mu * (1 - a_mu)) / a_pc;
            end
            % Occasionally explore based solely on posterior variance
            if (rand() < self.exp_rate)
                [max_var max_idx] = max(a_vars);
                best_arm = max_idx;
            else
                % Find location of the "gap center" for these return samples
                ars = sort(a_returns,'descend');
                gap_loc = (ars(self.top_m) + ars(self.top_m+1)) / 2;
                % Compute normalized distance to gap center for each arm
                a_gaps = abs(a_returns - gap_loc) ./ sqrt(a_vars);
                [min_gap min_idx] = min(a_gaps);
                % Select arm with smallest normalized distance to gap center
                best_arm = min_idx;
            end  
            return
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % SIGNIFICANCE RELATED METHODS %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        function [group_conf arm_confs] = ...
                compute_significance(self, group, sample_count)
            % Check significance of the current set of trial results for the
            % given group.
            if ~exist('sample_count','var')
                sample_count = 2000;
            end
            sample_means = zeros(self.arm_count,sample_count);
            for a=1:self.arm_count,
                % Sample returns for each arm from a beta distribution with
                % parameters alpha=a_n and beta=b_n, where a_n and b_n are
                % maintained via Bayesian updates after each arm pull.
                as = self.bandit_stats(group,a);
                sample_means(a,:) = betarnd(as.a_n, as.b_n, 1, sample_count);
            end
            % topm_map is computed to hold an array containing the indices of
            % the top m arms according to sample-based MAP estimates. The
            % indices in topm_map appear ordered by index value.
            arm_returns = mean(sample_means,2);
            [ar_vals ar_idx] = sort(arm_returns,'descend');
            topm_map = [ar_idx(1:self.top_m); -1];
            topm_map = sort(topm_map,'ascend');
            % When checking for the top m > 1 arms, things get a bit complex.
            % We start by sorting the arm returns for each sample set, which
            % also provides the return-sorted arm indices.
            [ar_vals ar_idx] = sort(sample_means,'descend');
            % Extract the top m indices for each sample.
            topm_samp = ar_idx(1:self.top_m,:);
            topm_samp = [topm_samp; zeros(1,sample_count)];
            % Sort the top m indices for each sample, as for topm_map
            topm_samp = sort(topm_samp,'ascend');
            % Count the fraction of samples for which the MAP top m are the same
            % as the sampled top m.
            hit_counts = sum(bsxfun(@eq,topm_map,topm_samp));
            group_conf = sum(hit_counts == self.top_m) / sample_count;
            % Independently estimate the confidence of each arm's top-m-ness by
            % counting the number of times each arm appears among the sampled
            % top m.
            arm_confs = zeros(1,self.arm_count);
            for a=1:self.arm_count,
                p = numel(find(topm_samp == a)) / sample_count;
                arm_confs(a) = max(p, (1 - p));
            end
            return
        end
        
        function [sgs group_confs arm_confs] = get_significant_groups(self)
            % Get the set of groups that currently have a significant result,
            % and record the confidence for each group's result.
            group_confs = zeros(1,self.group_count);
            arm_confs = zeros(self.group_count,self.arm_count);
            for g=1:self.group_count,
                [g_conf a_confs] = self.compute_significance(g);
                group_confs(g) = g_conf;
                arm_confs(g,:) = a_confs;
                % Check the "group-wise" confidence and raise per-arm
                % significance thresholds if all arms have passed
                a_thresh = self.bandit_stats(g,1).sig_thresh;
                if (min(a_confs) > a_thresh)
                    new_thresh = 1 - (0.5 * (1 - a_thresh));
                    for a=1:self.arm_count,
                        self.bandit_stats(g,a).sig_thresh = new_thresh;
                    end
                end     
            end
            sgs = find(group_confs > self.sig_thresh);
            return
        end
        
        function [succ_probs] = compute_succ_probs(self)
            % Compute a union bound based frequentist bound on the probability
            % that the current empirical top_m set is correct.
            arm_pcs = self.get_pull_counts();
            arm_means = zeros(self.group_count,self.arm_count);
            for g=1:self.group_count,
                for a=1:self.arm_count,
                    arm_means(g,a) = self.get_arm_mean(g,a);
                end
            end
            fail_probs = zeros(1,self.group_count);
            for g=1:self.group_count
                ars = sort(arm_means(g,:),'descend');
                gap_loc = (ars(self.top_m) + ars(self.top_m+1)) / 2;
                gaps = abs(arm_means(g,:) - gap_loc);
                vars = arm_means(g,:) .* (1 - arm_means(g,:));
                pcs = arm_pcs(g,:);
                fail_probs(g) = ...
                    sum(1 - normcdf(gaps .* (sqrt(pcs) ./ sqrt(vars)), 0, 1));
            end
            succ_probs = max(0, 1-fail_probs);
            return
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % TRIAL MANAGEMENT METHODS %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        function [best_group] = run_trial(self, epsilon, group_confs)
            % Run a single trial with the managed MultiArmBandit. Use
            % epsilon-greedy group/arm selection. Group is selected first, by
            % Thompson-like sampling, and then an arm from within that group is
            % selected by Thompson-like sampling.
            if (rand() < epsilon)
                best_group = randi(self.group_count);
                best_arm = randi(self.arm_count);
            else
                best_group = self.pick_group(group_confs);
                best_arm = self.pick_arm(best_group, group_confs);
            end
            % Run a trial for the selected arm
            self.pull_arm(best_group,best_arm);
            return
        end
        
        function [results] = run_trials(self,...
                trial_rounds, init_rounds, epsilon, thresh_start, thresh_scale)
            % Run the given number of trial rounds, playing one arm per round.
            sample_count = 50;
            self.sig_thresh = thresh_start;
            group_confs = zeros(self.group_count, trial_rounds);
            group_sprobs = zeros(self.group_count, trial_rounds);
            arm_selects = zeros(trial_rounds, self.group_count, self.arm_count);
            group_gaps = zeros(trial_rounds, sample_count);
            group_sums = zeros(trial_rounds, sample_count);
            opt_costs = zeros(trial_rounds, sample_count);
            init_pulls = self.uniform_allocation(init_rounds);
            [sig_groups confs] = self.get_significant_groups();
            succ_probs = self.compute_succ_probs();
            for t_num=1:trial_rounds,
                % Run a trial (i.e. pull an arm)
                if (t_num <= init_rounds)
                    % Do a uniformly-allocated initialization trial
                    self.pull_arm(init_pulls(t_num,1),init_pulls(t_num,2));
                else
                    % Do a "selected" arm trial
                    self.run_trial(epsilon, confs);
                end
                if (mod(t_num,50) == 0 || t_num == 1)
                    % Compute Bayesian gap-sample-based cost estimate
                    [sig_groups confs] = self.get_significant_groups();
                    succ_probs = self.compute_succ_probs();
                    a_selects = zeros(self.group_count,self.arm_count);
                    g_gaps = zeros(1,sample_count);
                    g_sums = zeros(1,sample_count);
                    for s=1:sample_count,
                        [best_group best_gap best_sum] = self.pick_group(confs);
                        g_gaps(s) = best_gap;
                        g_sums(s) = best_sum;
                        best_arm = self.pick_arm(best_group, confs);
                        a_selects(best_group, best_arm) = ...
                            a_selects(best_group, best_arm) + (1 / sample_count);
                    end
                    o_costs = StaticTopMOpt.ucb_comp_cost(...
                        self.bandit_stats, 1, self.top_m, sample_count);
                    if (mod(t_num, 500) == 0)
                        % Display the current trial and confidence
                        fprintf('%4.d: %.4f\n', t_num, mean(succ_probs));
                    end
                end
                % Store information about this round
                arm_selects(t_num,:,:) = a_selects ./ sample_count;
                group_gaps(t_num,:) = g_gaps;
                group_sums(t_num,:) = g_sums;
                opt_costs(t_num,:) = o_costs;
                group_confs(:,t_num) = confs;
                group_sprobs(:,t_num) = succ_probs;
            end
            results = struct();
            results.group_confs = group_confs;
            results.group_sprobs = group_sprobs;
            results.arm_selects = arm_selects;
            results.group_gaps = group_gaps;
            results.group_sums = group_sums;
            results.opt_costs = opt_costs;
            return
        end
        
        
        function [results] = ...
                run_slim_trials( self, trial_rounds, init_rounds, epsilon )
            % Run the given number of trial rounds, playing one arm per round.
            function [ confs ] = get_group_confs( sample_count )
                % Compute set of group confidences
                confs = zeros(self.group_count,1);
                for g=1:self.group_count,
                    confs(g) = self.compute_significance(g, sample_count);
                end
                return
            end
            group_confs = zeros(self.group_count, trial_rounds);
            group_sprobs = zeros(self.group_count, trial_rounds);
            init_pulls = self.uniform_allocation(init_rounds);
            g_confs = get_group_confs(1000);
            g_sprobs = self.compute_succ_probs();
            for t_num=1:trial_rounds,
                % Run a trial (i.e. pull an arm)
                if (t_num <= init_rounds)
                    % Do a uniformly-allocated initialization trial
                    self.pull_arm(init_pulls(t_num,1),init_pulls(t_num,2));
                else
                    % Do a "selected" arm trial
                    self.run_trial(epsilon, g_confs);
                end
                % Compute the set of significant groups (sometimes)
                if (mod(t_num,20) == 0)
                    g_confs = get_group_confs(1000);
                    g_sprobs = self.compute_succ_probs();
                end
                group_confs(:,t_num) = g_confs;
                group_sprobs(:,t_num) = g_sprobs;
                % Display the group confidences
                if (mod(t_num, 100) == 0)
                    fprintf('%4.d: %.4f\n', t_num,mean(g_confs));
                end
            end
            results = struct();
            results.group_confs = group_confs;
            results.group_sprobs = group_sprobs;
            return
        end
        
    end % END INSTANCE METHODS
    
end







%%%%%%%%%%%%%%
% EYE BUFFER %
%%%%%%%%%%%%%%

