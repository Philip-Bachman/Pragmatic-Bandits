classdef StaticTopMOpt < handle
    % StaticTopMOpt works a MultiArmBandit of type 'bernoulli'
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
    end
    
    methods
        function [self] = StaticTopMOpt(ma_bandit, top_m, a_0, b_0)
            % Constructor for multiarmed bandit StaticTopMOpt 
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
        
        function [best_group] = pick_group(self, group_results, group_confs)
            % Pick a group to pull from the managed MultiArmBandit. Ignore the
            % groups in sig_groups (which are already assumed confident).
            if (self.group_count == 1)
                best_group = 1;
                return
            end
            group_costs = zeros(1,self.group_count);
            for g=1:self.group_count,
                group_costs(g) = ...
                    find(group_results(g).results.fail_probs > 0.95,1,'first');
            end
            group = randsample(self.group_count,1,true,group_costs.^(-1));
            while (group_confs(group) > self.sig_thresh)
                group = randsample(self.group_count,1,true,group_costs.^(-1));
            end
            best_group = group;
            return
        end
        
        function [best_arm] = pick_arm(self, group, group_results)
            % Pick an arm to pull from the managed MultiArmBandit, given the
            % group from which an arm is to be picked. 
            arm_pcs = group_results(group).results.pull_counts(end,:);
            best_arm = randsample(self.arm_count,1,true,arm_pcs);
            return
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % SIGNIFICANCE RELATED METHODS %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        function [group_conf arm_confs] = compute_significance(self, group)
            % Check significance of the current set of trial results for the
            % given group.
            sample_count = 2000;
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
        
        function [best_group] = ...
                run_trial(self, epsilon, group_results, group_confs)
            % Run a single trial with the managed MultiArmBandit. Use
            % epsilon-greedy group/arm selection.
            if (rand() < epsilon)
                best_group = randi(self.group_count);
                best_arm = randi(self.arm_count);
            else
                best_group = self.pick_group(group_results, group_confs);
                best_arm = self.pick_arm(best_group, group_results);
            end
            % Run a trial for the selected arm
            self.pull_arm(best_group,best_arm);
            return
        end
        
        function [results] = run_trials(self,...
                trial_rounds, init_rounds, epsilon, thresh_start, thresh_scale)
            % Run the given number of trial rounds, playing one arm per round.
            self.sig_thresh = thresh_start;
            sig_counts = zeros(1,trial_rounds);
            select_accs = zeros(1,trial_rounds);
            group_confs = zeros(self.group_count, trial_rounds);
            group_sprobs = zeros(self.group_count, trial_rounds);
            init_pulls = self.uniform_allocation(init_rounds);
            [sig_groups confs] = self.get_significant_groups();
            succ_probs = self.compute_succ_probs();
            % Optimate the managed bandit
            fprintf('Optimating bandit...');
            group_inits = round(init_rounds/self.group_count);
            group_results = struct();
            for g=1:self.group_count,
                group_results(g).results = StaticTopMOpt.optimate_bandit(...
                    self.bandit, g, self.top_m, group_inits, group_inits);
            end
            fprintf('done!\n');
            for t_num=1:trial_rounds,
                % Run a trial (i.e. pull an arm)
                if (t_num <= init_rounds)
                    % Do a uniformly-allocated initialization trial
                    self.pull_arm(init_pulls(t_num,1),init_pulls(t_num,2));
                else
                    % Do a "selected" arm trial
                    self.run_trial(epsilon, group_results, confs);
                end
                % Compute the set of significant groups (sometimes)
                if (mod(t_num,10) == 0)
                    [sig_groups confs] = self.get_significant_groups();
                    succ_probs = self.compute_succ_probs();
                    % If all groups are significant, raise confidence threshold
                    if (numel(sig_groups) == self.group_count)
                        fprintf('    raising sig_thresh\n');
                        thresh_gap = (1 - self.sig_thresh) * thresh_scale;
                        self.sig_thresh = 1 - thresh_gap;
                        [sig_groups confs]=self.get_significant_groups();
                    end
                end
                if (mod(t_num,200) == 0)
                    fprintf('Optimating...');
                    for g=1:self.group_count,
                        group_results(g).results = ...
                            StaticTopMOpt.optimate_bayes( self.bandit_stats,...
                            g, self.top_m, trial_rounds, group_inits, 10 );
                    end
                    fprintf('done.\n');
                end
                group_confs(:,t_num) = confs;
                group_sprobs(:,t_num) = succ_probs;
                sig_counts(t_num) = numel(sig_groups);
                best_arms = self.get_best_arms(self.top_m);
                good_arms = sum(best_arms(:) <= self.top_m);
                select_accs(t_num) = good_arms / (self.group_count*self.top_m);
                % Display the current number of significant groups and the
                % average group confidence.
                if (mod(t_num, 50) == 0)
                    fprintf('%4.d: (%d, %.4f, %.4f)\n', ...
                        t_num,numel(sig_groups),mean(confs),select_accs(t_num));
                end
            end
            results = struct();
            results.sig_counts = sig_counts;
            results.group_confs = group_confs;
            results.group_sprobs = group_sprobs;
            results.select_accs = select_accs;
            results.group_results = group_results;
            return
        end
    end % END INSTANCE METHODS
    
    methods (Static = true)
        
        function [comp_costs] = ucb_comp_cost(stats, group, top_m, sample_count)
            % Compute a variance-included compexity cost like gabillon et. al
            a_count = size(stats,2);
            comp_costs = zeros(1,sample_count);
            for s=1:sample_count,
                a_returns = zeros(1, a_count);
                a_vars = zeros(1, a_count);
                for a=1:a_count,
                    r = betarnd(stats(group,a).a_n,stats(group,a).b_n);
                    a_returns(a) = r;
                    a_vars(a) = r * (1 - r);
                end
                [a_returns a_idx] = sort(a_returns,'descend');
                a_vars = a_vars(a_idx);
                % Compute the UCB-ish complexity estimate for this bandit
                a_comps = zeros(1,a_count);
                for a=1:a_count,
                    if (a <= top_m)
                        a_gap = a_returns(a) - a_returns(top_m + 1);
                    else
                        a_gap = a_returns(top_m) - a_returns(a);
                    end
                    a_var = a_vars(a);
                    a_comps(a) = ...
                        (sqrt(a_var) + sqrt(a_var + (16/3)*a_gap))^2 / a_gap^2;
                end
                comp_costs(s) = sum(a_comps);
            end
            return
        end
        
        function [comp_costs] = map_comp_cost(stats, group, top_m, sample_count)
            % Compute a variance-included compexity cost like gabillon et. al
            a_count = size(stats,2);
            comp_costs = zeros(1,sample_count);
            a_returns = zeros(1, a_count);
            a_vars = zeros(1, a_count);
            for a=1:a_count,
                r = stats(group,a).a_n / (stats(group,a).a_n + stats(group,a).b_n);
                a_returns(a) = r;
                a_vars(a) = r * (1 - r);
            end
            [a_returns a_idx] = sort(a_returns,'descend');
            a_vars = a_vars(a_idx);
            % Compute the UCB-ish complexity estimate for this bandit
            a_comps = zeros(1,a_count);
            for a=1:a_count,
                if (a <= top_m)
                    a_gap = a_returns(a) - a_returns(top_m + 1);
                else
                    a_gap = a_returns(top_m) - a_returns(a);
                end
                a_var = a_vars(a);
                a_comps(a) = ...
                    (sqrt(a_var) + sqrt(a_var + (16/3)*a_gap))^2 / a_gap^2;
            end
            comp_costs(1:end) = sum(a_comps);
            return
        end
 
        
        function [ results ] = ...
                optimate_stats(stats, group, top_m, trial_rounds, init_rounds)
            % Compute means and variances for the given group of the given
            % multi-armed bandit stats. Assume bernoulli banditry.
            a_count = size(stats,2);
            a_returns = zeros(1, a_count);
            a_vars = zeros(1, a_count);
            for a=1:a_count,
                r = mean(stats(group,a).pulls);
                a_returns(a) = r;
                a_vars(a) = r * (1 - r);
            end
            results = StaticTopMOpt.optimate(...
                a_returns, a_vars, top_m, trial_rounds, init_rounds);
            return
        end
        
        function [ results ] = optimate_bayes(...
                stats, group, top_m, trial_rounds, init_rounds, sample_count)
            % Compute means and variances for the given group of the given
            % multi-armed bandit stats. Assume bernoulli banditry.
            a_count = size(stats,2);
            pull_counts = zeros(trial_rounds,a_count);
            fail_probs = zeros(trial_rounds,1);
            all_fail_probs = zeros(sample_count,trial_rounds);
            for s=1:sample_count,
                a_returns = zeros(1, a_count);
                a_vars = zeros(1, a_count);
                for a=1:a_count,
                    r = betarnd(stats(group,a).a_n,stats(group,a).b_n);
                    a_returns(a) = r;
                    a_vars(a) = r * (1 - r);
                end
                s_res = StaticTopMOpt.optimate(...
                    a_returns, a_vars, top_m, trial_rounds, init_rounds);
                pull_counts = pull_counts + (s_res.pull_counts ./ sample_count);
                fail_probs = fail_probs + (s_res.fail_probs ./ sample_count);
                all_fail_probs(s,:) = fail_probs';
            end
            results = struct();
            results.pull_counts = pull_counts;
            results.fail_probs = fail_probs;
            results.all_fail_probs = all_fail_probs;
            return
        end
        
        function [ results ] = ...
                optimate_bandit(bandit, group, top_m, trial_rounds, init_rounds)
            % Compute means and variances for the given group of the given
            % multi-armed bandit. Assume (unnecessarily) bernoulli banditry.
            a_returns = zeros(1, bandit.arm_count);
            a_vars = zeros(1, bandit.arm_count);
            for a=1:bandit.arm_count,
                r = bandit.arm_groups(group,a).return;
                a_returns(a) = r;
                a_vars(a) = r * (1 - r);
            end
            results = StaticTopMOpt.optimate(...
                a_returns, a_vars, top_m, trial_rounds, init_rounds);
            return
        end

        function [ results ] = ...
                optimate( a_means, a_vars, top_m, trial_rounds, init_rounds )
            % Compute an "optimal" trial allocation policy for the bandit whose
            % arms have means "a_means" and variances "a_vars". The objective of
            % the policy is to minimize the number of trials required for
            % identifying the top_m best arms with high confidence.
            %
            % Parameters:
            %   a_means: the mean of each arm in the bandit to optimalize
            %   a_vars: the variance of each arm in the bandit to optimize
            %   top_m: the number of arms to target for selection
            %   trial_rounds: the number of trials to consider allocating
            %   init_rounds: the number of rounds to allocate uniformly
            % Outputs:
            %   results.pull_counts: the cumulative pull count for each arm in
            %                        each round.
            %   results.fail_probs: the estimated failure probability per round
            %    
            a_count = numel(a_means);
            % Compute the gap location and per-arm gaps for the given bandit
            as = sort(a_means,'descend');
            gap_loc = (as(top_m) + as(top_m+1)) / 2;
            a_gaps = abs(a_means - gap_loc);
            % Compute union bound estimates of failure probability at each round
            pull_counts = zeros(trial_rounds,a_count);
            fail_probs = zeros(trial_rounds,1);
            prev_pc = zeros(1,a_count);
            for i=1:trial_rounds,
                if (i <= init_rounds)
                    % For init rounds, assume (cyclic) uniform allocation
                    a = mod(i-1, a_count) + 1;
                    min_pc = prev_pc;
                    min_pc(a) = min_pc(a) + 1;
                    new_pts = a_gaps .* (sqrt(min_pc) ./ sqrt(a_vars));
                    min_fp = sum(1 - normcdf(new_pts, 0, 1));
                else
                    % For other rounds, allocate to the arm which maximally
                    % reduces the union bound estimate of failure probability
                    new_pcs = bsxfun(@plus, eye(a_count), prev_pc);
                    new_pts = bsxfun(@times, a_gaps,...
                        bsxfun(@rdivide, sqrt(new_pcs), sqrt(a_vars)));
                    new_fps = sum(1 - normcdf(new_pts, 0, 1), 2);
                    [min_fp min_idx] = min(new_fps);
                    min_pc = new_pcs(min_idx,:);
                end
                % Record the minimum pull counts and failure prob, then reset 
                % prev values for use in next round.
                pull_counts(i,:) = min_pc;
                fail_probs(i) = min_fp;
                prev_pc = min_pc;
            end
            % Pack the results structure for returnage
            results = struct();
            results.pull_counts = pull_counts;
            results.fail_probs = fail_probs;
            return
        end
        
    end % END STATIC METHODS
    
end







%%%%%%%%%%%%%%
% EYE BUFFER %
%%%%%%%%%%%%%%

