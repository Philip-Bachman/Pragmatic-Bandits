classdef TopMNOpt < handle
    % TopMNOpt works a MultiArmBandit of type 'normal'
    % This class performs 'most successful tests' optimization of a
    % MultiArmBandit instance of type 'normal'. The objective, for a given
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
        % alpha is alpha for initial gamma prior over arm precisions
        alpha
        % beta is beta for initial gamma prior over arm precisions
        beta
        % kappa is kappa for initial normal-gamma prior over means/precisions
        kappa
        % sig_thresh is the threshold at which a group's result is significant
        sig_thresh
        % top_m is the number of top arms to try and select
        top_m
        ap_samples
    end
    
    methods
        function [self] = TopMNOpt(ma_bandit, top_m, a_0, b_0, k_0)
            % Constructor for multiarmed bandit TopMNOpt
            if ~exist('a_0','var')
                a_0 = 2.0;
            end
            if ~exist('b_0','var')
                b_0 = 2.0;
            end
            if ~exist('k_0','var')
                k_0 = 2.0;
            end
            self.alpha = a_0;
            self.beta = b_0;
            self.kappa = k_0;
            self.sig_thresh = 0.99;
            self.set_bandit(ma_bandit);
            self.top_m = top_m;
            self.ap_samples = 1;
            return
        end
        
        function [self] = set_bandit(self, ma_bandit)
            % Set the bandit instance for this NormalsOptimizer to optimize.
            self.bandit = ma_bandit;
            self.bandit.reset_arms(0);
            % initialize optimization structures
            self.group_count = ma_bandit.group_count;
            self.arm_count = ma_bandit.arm_count;
            self.bandit_stats = struct();
            for g=1:self.group_count,
                for a=1:self.arm_count,
                    self.bandit_stats(g,a).pulls = [];
                    self.bandit_stats(g,a).sig_thresh = 0.5;
                    self.bandit_stats(g,a).m_0 = 0.0;
                    self.bandit_stats(g,a).a_0 = self.alpha;
                    self.bandit_stats(g,a).b_0 = self.beta;
                    self.bandit_stats(g,a).k_0 = self.kappa;
                    self.bandit_stats(g,a).m_n = 0.0;
                    self.bandit_stats(g,a).a_n = self.alpha;
                    self.bandit_stats(g,a).b_n = self.beta;
                    self.bandit_stats(g,a).k_n = self.kappa;
                end
            end
            return
        end
        
        function [arm_return] = pull_arm(self, group, arm)
            % Pull the given arm in the managed bandit and record stats
            arm_return = self.bandit.pull_arm(group,arm);
            arm_stats = self.bandit_stats(group,arm);
            arm_stats.pulls(end+1) = arm_return;
            % Update normal-gamma distribution parameters for this arm
            pull_count = numel(arm_stats.pulls);
            pull_mu = mean(arm_stats.pulls);
            a_n = arm_stats.a_0 + (pull_count / 2);
            b_n = arm_stats.b_0 + ...
                0.5 * (sum((arm_stats.pulls - pull_mu).^2)) + ...
                ((arm_stats.k_0 * pull_count)*(pull_mu - arm_stats.m_0)^2) / ...
                (2 * (arm_stats.k_0 + pull_count));
            k_n = arm_stats.k_0 + pull_count;
            m_n = ((arm_stats.m_0*arm_stats.k_0) + (pull_count*pull_mu)) / ...
                (arm_stats.k_0 + pull_count);
            % Set arm_stats m_n, k_n, a_b, b_n...
            arm_stats.a_n = a_n;
            arm_stats.b_n = b_n;
            arm_stats.k_n = k_n;
            arm_stats.m_n = m_n;
            self.bandit_stats(group,arm) = arm_stats;
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
            % normal-gamma beliefs. Use a sample-based approximation of the
            % belief-based mean return for each arm.
            if ~exist('top_m','var')
                top_m = self.top_m;
            end
            best_arms = zeros(self.group_count,top_m);
            best_returns = zeros(self.group_count,top_m);
            for g=1:self.group_count,
                ars = zeros(1,self.arm_count);
                for a=1:self.arm_count,
                    ars(a) = self.get_arm_mean(g,a);
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
            sample_means = TopMNOpt.sample_ng_post(as, 10000);
            arm_mean = mean(sample_means);
            return
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % GROUP AND ARM PICKING METHODS %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        function [best_group] = pick_group(self, group_confs)
            % Pick a group to pull from the managed MultiArmBandit. Ignore the
            % groups which already have confidence > self.sig_thresh.
            sig_groups = find(group_confs > self.sig_thresh);
            arm_mus = zeros(self.group_count,self.arm_count);
            arm_vars = zeros(self.group_count,self.arm_count);
            for g=1:self.group_count,
                for a=1:self.arm_count,
                    as = self.bandit_stats(g,a);
                    lambda = gamrnd(as.a_n,1/as.b_n);
                    mu_var = 1 / (as.k_n * lambda);
                    mu = (randn() * sqrt(mu_var)) + as.m_n;
                    arm_mus(g,a) = mu;
                    arm_vars(g,a) = 1 / lambda;
                    %arm_vars(g,a) = mu_var;
                end
            end
            best_group = randi(self.group_count);
            best_gap = 0;
            for g=1:self.group_count,
                % For group, get the 'approximate' normalized gap between its
                % m and m+1 arms, where the difference in means is normalized to
                % account for differences in variance of the mean estimates.
                [g_mus g_idx] = sort(arm_mus(g,:),'descend');
                g_vars = arm_vars(g,g_idx);
                g_gap = (g_mus(self.top_m) - g_mus(self.top_m+1)) / ...
                    sqrt(g_vars(self.top_m) + g_vars(self.top_m+1));
                % Pick not-yet-significant group with largest normalized gap
                if ((g_gap > best_gap) && ~ismember(g,sig_groups))
                    best_gap = g_gap;
                    best_group = g;
                end
            end
            return
        end
        
%         function [best_arm] = pick_arm(self, group, arm_confs)
%             % Pick an arm to pull from the managed MultiArmBandit, given the
%             % group from which an arm is to be picked.
%             arm_mus = zeros(1,self.arm_count);
%             arm_igs = zeros(1,self.arm_count);
%             for a=1:self.arm_count,
%                 as = self.bandit_stats(group,a);
%                 pc = numel(as.pulls);
%                 lambda = gamrnd(as.a_n, 1/as.b_n);
%                 mu_var = 1 / (as.k_n * lambda);
%                 mu = (randn() * sqrt(mu_var)) + as.m_n;
%                 arm_mus(a) = mu;
%                 arm_igs(a) = ((1/lambda)/pc) - ((1/lambda)/(pc+1));
%             end
%             [arm_mus arm_idx] = sort(arm_mus,'descend');
%             arm_igs = arm_igs(arm_idx);
%             if (arm_igs(self.top_m) >= arm_igs(self.top_m+1))
%                 best_arm = arm_idx(self.top_m);
%             else
%                 best_arm = arm_idx(self.top_m+1);
%             end
%             return
%         end
        
        function [best_arm] = pick_arm(self, group, arm_confs)
            % Pick an arm to pull from the managed MultiArmBandit, given the
            % group from which an arm is to be picked.
            arm_mus = zeros(1,self.arm_count);
            arm_vars = zeros(1,self.arm_count);
            for a=1:self.arm_count,
                as = self.bandit_stats(group,a);
                pc = numel(as.pulls);
                lambda = gamrnd(as.a_n, 1/as.b_n);
                mu_var = 1 / (as.k_n * lambda);
                mu = (randn() * sqrt(mu_var)) + as.m_n;
                arm_mus(a) = mu;
                arm_vars(a) = mu_var / pc;
            end
            [arm_mus arm_idx] = sort(arm_mus,'descend');
            arm_vars = arm_vars(arm_idx);
            gap_loc = (arm_mus(self.top_m) + arm_mus(self.top_m+1)) / 2;
            arm_gaps = abs(arm_mus - gap_loc) ./ sqrt(arm_vars);
            [min_gap min_idx] = min(arm_gaps);
            best_arm = arm_idx(min_idx);
            return
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % SIGNIFICANCE RELATED METHODS %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        function [group_conf arm_confs] = compute_significance(self, group)
            % Check significance of the current set of trial results for the
            % given group.
            sample_count = 10000;
            sample_means = zeros(self.arm_count,sample_count);
            for a=1:self.arm_count,
                % Sample means and variances for each arm from a normal-gamma
                % distribution with parameters alpha=a_n, beta=b_n, mu=m_n, and
                % kappa=k_n, where the values a_n, b_n, m_n, and k_n are
                % maintained via Bayesian updates after each arm pull and kept
                % in self.bandit_stats(group, arm) for each group/arm.
                arm_stats = self.bandit_stats(group,a);
                arm_means = TopMNOpt.sample_ng_post(arm_stats, sample_count);
                sample_means(a,:) = arm_means;
            end
            % topm_map is computed to hold an array containing the indices of
            % the top m arms according to sample-based MAP estimates. The
            % indices in topm_map appear ordered by index value.
            arm_returns = mean(sample_means,2);
            [ar_vals ar_idx] = sort(arm_returns,'descend');
            topm_map = [ar_idx(1:self.top_m); -1];
            topm_map = sort(topm_map,'ascend');
            % Sort the arm returns for each sample set, using multi-dimensional
            % array sort, which also returns the return-ordered indices for each
            % sample.
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
                if (self.ap_samples < 2)
                    best_arm = self.pick_arm(best_group, group_confs);
                else
                    best_arms = zeros(1,self.ap_samples);
                    for i=1:self.ap_samples,
                       best_arms(i) = self.pick_arm(best_group, group_confs);
                    end
                    best_arm = randsample(best_arms,1);
                end
            end
            % Run a trial for the selected arm
            self.pull_arm(best_group,best_arm);
            return
        end
        
        function [sig_counts trial_confs] = run_trials(self,...
                trial_rounds, init_rounds, epsilon, thresh_start, thresh_scale)
            % Run the given number of trial rounds, playing one arm per round.
            self.sig_thresh = thresh_start;
            sig_counts = zeros(trial_rounds,1);
            trial_confs = zeros(self.group_count, trial_rounds);
            all_confs = zeros(self.group_count, self.arm_count, trial_rounds);
            all_means = zeros(self.group_count, self.arm_count, trial_rounds);
            init_pulls = self.uniform_allocation(init_rounds);
            [sig_groups group_confs arm_confs] = ...
                self.get_significant_groups();
            t_sig = -10;
            for t_num=1:trial_rounds,
                % Run a trial (i.e. pull an arm)
                if (t_num <= init_rounds)
                    % Do a uniformly-allocated initialization trial
                    self.pull_arm(init_pulls(t_num,1),init_pulls(t_num,2));
                else
                    % Do a "selected" arm trial
                    self.run_trial(epsilon, group_confs);
                end
                % Compute the set of significant groups (sometimes)
                if (t_num - t_sig > 10)
                    [sig_groups group_confs arm_confs] = ...
                        self.get_significant_groups();
                    t_sig = t_num;
                    % If all groups are significant, raise self.sig_thresh
                    if (numel(sig_groups) == self.group_count)
                        thresh_gap = (1 - self.sig_thresh) * thresh_scale;
                        self.sig_thresh = 1 - thresh_gap;
                        fprintf('    raising sig_thresh\n');
                        [sig_groups group_confs arm_confs] = ...
                            self.get_significant_groups();
                    end
                end
                trial_confs(:,t_num) = group_confs;
                all_confs(:,:,t_num) = arm_confs;
                sig_count = numel(sig_groups);
                sig_counts(t_num) = sig_count;
                % Display the current number of significant groups and the
                % average group confidence.
                if (mod(t_num, 25) == 0)
                    fprintf('%4.d: (%d, %.4f)\n', ...
                        t_num, sig_count, mean(group_confs));
                end
                for g=1:self.group_count,
                    for a=1:self.arm_count,
                        pulls = self.bandit_stats(g,a).pulls;
                        if (numel(pulls) > 0)
                            all_means(g,a,t_num) = mean(pulls);
                        end
                    end
                end
            end
            return
        end
        
        function [sig_counts trial_confs] = ...
                run_uniform_trials(self, trial_rounds, sig_thresh)
            % Run the given number of trial rounds, playing all arms in a
            % selected group at each trial round. Use uniform allocation of
            % trials, e.g. cycle over the groups.
            if exist('sig_thresh','var')
                self.sig_thresh = sig_thresh;
            end
            sig_counts = zeros(trial_rounds,1);
            trial_confs = zeros(self.group_count, trial_rounds);
            t_sig = -10;
            group = 1;
            arm = 1;
            for t_num=1:trial_rounds,
                % Pull an arm, determined by group/arm allocation in pull_idx
                self.pull_arm(group,arm);
                [group arm] = self.uniform_step(group, arm);
                if (t_num - t_sig > 10)
                    [sig_groups group_confs arm_confs] = ...
                        self.get_significant_groups();
                    t_sig = t_num;
                end
                trial_confs(:,t_num) = group_confs;
                sig_count = numel(sig_groups);
                sig_counts(t_num) = sig_count;
                if (mod(t_num, 25) == 0)
                    fprintf('%4.d: (%d, %.4f)\n', ...
                        t_num, sig_count, mean(group_confs));
                end
            end
            return
        end
        
    end % END INSTANCE METHODS
    
    methods (Static = true)
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % NORMAL-GAMMA SAMPLING STUFF %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        function [means vars] = sample_ng_post(a_stats, sample_count)
            % Sample means and variances from the normal-gamma posterior made
            % by the observations for a bandit with stats struct() a_stats.
            lams = gamrnd(a_stats.a_n, 1/a_stats.b_n, 1, sample_count);
            mus = randn(1, sample_count) .* sqrt((a_stats.k_n * lams).^(-1));
            means = mus + a_stats.m_n;
            vars = lams.^(-1);
            return
        end
        
        function [ x ] = sample_ng_vals(a_stats, sample_count)
            % Sample returns from the normal-gamma distribution currently
            % estimated for a bandit with stats a_stats.
            [means vars] = TopMNOpt.sample_ng_post(a_stats,sample_count);
            x = (randn(1,sample_count) .* sqrt(vars)) + means;
            return
        end
        
        function [new_stats] = sample_pull_arm(a_stats, sample_count)
            % Sample many single pull->updates of the given arm, using its
            % estimated normal-gamma distribution. For each sample, compute the
            % resulting updated normal-gamma distribution/parameters.
            as = a_stats;
            % Sample pulls from the arm given its current normal-gamma
            a_pulls = TopMNOpt.sample_ng_vals(as, sample_count);
            new_stats = struct();
            % Perform "fake" updates for each of the sample pulls
            for i=1:sample_count,
                % Create the pull set for this sampled arm update
                n_pulls = [as.pulls a_pulls(i)];
                % Compute new normal-gamma distribution parameters
                pc = numel(n_pulls);
                pm = mean(n_pulls);
                a_n = as.a_0 + (pc / 2);
                b_n = as.b_0 + (0.5 * (sum((n_pulls - pm).^2))) + ...
                    (((as.k_0 * pc) * (pm - as.m_0)^2) / (2 * (as.k_0 + pc)));
                k_n = as.k_0 + pc;
                m_n = ((as.m_0 * as.k_0) + (pc * pm)) / (as.k_0 + pc);
                % Set fake updated arm_stats
                new_stats(i).pulls = n_pulls;
                new_stats(i).a_n = a_n;
                new_stats(i).b_n = b_n;
                new_stats(i).k_n = k_n;
                new_stats(i).m_n = m_n;
                new_stats(i).a_0 = as.a_0;
                new_stats(i).b_0 = as.b_0;
                new_stats(i).k_0 = as.k_0;
                new_stats(i).m_0 = as.m_0;
            end
            return
        end
        
        function [gain] = sample_pull_gain(a_stats, sample_count)
            % Compute a sample-based estimate of the expected "information gain"
            % for pulling a bandit arm with the stats a_stats.
            pre_means = TopMNOpt.sample_ng_post(a_stats,sample_count^2);
            new_stats = TopMNOpt.sample_pull_arm(a_stats, sample_count);
            post_means = [];
            for i=1:sample_count,
                as = new_stats(i);
                means = TopMNOpt.sample_ng_post(as, sample_count);
                post_means = [post_means means];
            end
            gain = var(pre_means) - var(post_means);
            return
        end
    end % END STATIC METHODS
end







%%%%%%%%%%%%%%
% EYE BUFFER %
%%%%%%%%%%%%%%

