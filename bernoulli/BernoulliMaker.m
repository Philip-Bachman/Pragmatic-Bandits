classdef BernoulliMaker < handle
    % BernoulliMaker makes bernoulli bandits with various arm distributions.
    % 
    
    properties
        % group_count is the number of arm groups for this bandit
        group_count
        % arm_count is the number of arms per arm group for this bandit
        arm_count
        % top_m gives the size of top group for each bandit produced
        top_m
        % return_type gives the return distribution: 'bernoulli' or 'normal'.
        return_type
    end
    
    methods
        function [self] = BernoulliMaker(group_count, arm_count, top_m)
            % Constructor for a bernoulli bandit maker
            self.group_count = group_count;
            self.arm_count = arm_count;
            self.top_m = top_m;
            self.return_type = 'bernoulli';
            return
        end
        
        function [a_returns gap] = get_returns_gap(self)
            % Get a set of sorted randomly selected returns and the gap between
            % the m best returns and the remaining returns.
            a_returns = rand(1,self.arm_count);
            a_returns = sort(a_returns,'descend');
            gap = a_returns(self.top_m) - a_returns(self.top_m+1);
            return
        end
        
        function [bandit] = distro_1(self)
            % Make a bernoulli bandit and set its arm returns. This is the
            % simplest return distribution; each arm's return is set by rand().
            bandit = MultiArmBandit(...
                self.group_count, self.arm_count, 'bernoulli');
            for g=1:bandit.group_count,
                for a=1:bandit.arm_count,
                   bandit.arm_groups(g,a).return = rand(); 
                end
            end
            bandit.reset_arms(1);
            return
        end
        
        function [bandit] = distro_2(self, easy_gap, hard_gap, hard_rate)
            % Make a bernoulli bandit and set its arm returns.
            bandit = MultiArmBandit(...
                self.group_count, self.arm_count, 'bernoulli');
            easy_count = round(self.group_count * (1 - hard_rate));
            for g=1:bandit.group_count,
                if (g > easy_count)
                    b_gap = hard_gap;
                else
                    b_gap = easy_gap;
                end
                bandit.arm_groups(g,1).return = 0.5 + b_gap;
                a_returns = rand(1,bandit.arm_count-1) .* 0.5;
                a_returns = a_returns + (0.5 - max(a_returns));
                a_returns = sort(a_returns,'descend');
                for a=2:bandit.arm_count,
                    bandit.arm_groups(g,a).return = a_returns(a-1);
                end
            end
            % Reset group order, to be from least to most complex
            g_costs = bandit.compute_ucb_costs(1);
            [g_costs g_idx] = sort(g_costs,'ascend');
            b_returns = zeros(self.group_count, self.arm_count);
            for g=1:bandit.group_count,
                for a=1:bandit.arm_count,
                    b_returns(g,a) = bandit.arm_groups(g_idx(g),a).return;
                end
            end
            for g=1:bandit.group_count,
                for a=1:bandit.arm_count,
                    bandit.arm_groups(g,a).return = b_returns(g,a);
                end
            end
            bandit.reset_arms(1);
            return
        end
        
        function [bandit] = distro_3(self, easy_range, hard_range, hard_rate)
            % Make a bernoulli bandit and set its arm returns. For this
            % distribution, each group is either easy or hard, with the expected
            % fraction of hard groups set by hard_rate. For easy groups, the m
            % best arms and the remaining arms is guaranteed to be greater than
            % easy_gap. For hard groups, the gap between the m best arms and
            % the remaining arms is guaranteed to be less than hard_gap.
            scale = 0.8;
            easy_range = easy_range ./ scale;
            hard_range = hard_range ./ scale;
            bandit = MultiArmBandit(...
                self.group_count, self.arm_count, 'bernoulli');
            easy_count = round(self.group_count * (1 - hard_rate));
            for g=1:bandit.group_count,
                if (g > easy_count)
                    % Pick an arm return distribution such that the difference
                    % between its largest and second largest returns is less
                    % than hard_gap.
                    [a_returns gap] = self.get_returns_gap();
                    while ((gap > hard_range(2)) || gap < hard_range(1))
                        [a_returns gap] = self.get_returns_gap();
                    end
                    a_returns = (a_returns .* scale) + ((1 - scale) / 2);
                    for a=1:bandit.arm_count,
                        bandit.arm_groups(g,a).return = a_returns(a);
                    end
                else
                    % Pick an arm return distribution such that the difference
                    % between its largest and second largest returns is more
                    % than easy_gap.
                    [a_returns gap] = self.get_returns_gap();
                    while ((gap > easy_range(2)) || gap < easy_range(1))
                        [a_returns gap] = self.get_returns_gap();
                    end
                    a_returns = (a_returns .* scale) + ((1 - scale) / 2);
                    for a=1:bandit.arm_count,
                        bandit.arm_groups(g,a).return = a_returns(a);
                    end
                end     
            end
            bandit.reset_arms(1);
            return
        end
        
        function [bandit] = distro_4(self, min_gap, max_gap, a1_rate)
            % Make a bernoulli bandit and set its arm returns. For this
            % distribution, each group has either arm 1 or some other arm as its
            % best arm. round(a1_rate * self.group_count) bandits will have arm
            % 1 as the best arm. Each bandit will have a gap of at least
            % 'min_gap' and at most 'max_gap' between its two best arms.
            bandit = MultiArmBandit(...
                self.group_count, self.arm_count, 'bernoulli');
            a1_count = round(self.group_count * a1_rate);
            for g=1:bandit.group_count,
                % Pick an arm return distribution such that the difference
                % between its best and second best returns is at least min_gap
                % and at most max_gap.
                gap = @( x ) x(1) - x(2);
                a_returns = self.get_returns_gap();
                while (gap(a_returns) < min_gap || gap(a_returns) > max_gap)
                    a_returns = self.get_returns_gap();
                end
                % Pick an ordering to associate the each return with an arm
                ord = randperm(self.arm_count);
                if (g <= a1_count)
                    % Pick a random order conditioned on ord(1) == 1
                    while (ord(1) ~= 1)
                        ord = randperm(self.arm_count);
                    end
                else
                    % Pick a random order conditioned on ord(1) ~= 1
                    while (ord(1) == 1)
                        ord = randperm(self.arm_count);
                    end
                end
                % Use the ordering to assign a return to each arm
                for a=1:bandit.arm_count,
                    bandit.arm_groups(g,a).return = a_returns(ord(a));
                end
            end
            bandit.reset_arms(1);
            return
        end
        
        function [bandit] = distro_5(self, min_gap, max_gap)
            % Make a bernoulli bandit and set its arm returns. For this
            % distribution, each group has a gap of at least min gap and at most
            % max gap between its top m arms and the remaining arms.
            bandit = MultiArmBandit(...
                self.group_count, self.arm_count, 'bernoulli');
            for g=1:bandit.group_count,
                % Pick an arm return distribution such that the difference
                % between its top m arms and the rest is in [min_gap...max_gap]
                [a_returns gap] = self.get_returns_gap();
                while ((gap > max_gap) || (gap < min_gap))
                    [a_returns gap] = self.get_returns_gap();
                end
                % Use the ordering to assign a return to each arm
                for a=1:bandit.arm_count,
                    bandit.arm_groups(g,a).return = a_returns(a);
                end
            end
            bandit.reset_arms(1);
            return
        end
         
    end
    
    methods (Static = true)
        
        function [bandit] = mbbai_problem1()
            % Make a bernoulli bandit and set its arm returns to match those in
            % problem 1 from "Multi-bandit Best Arm Identification" by Gabillon,
            % Ghavamzadeh, Lazaric, and Bubeck (NIPS 2011).
            bandit = MultiArmBandit(2, 4, 'bernoulli');
            % Returns for group 1 are (0.5, 0.45, 0.4, 0.3)
            bandit.arm_groups(1,1).return = 0.5;
            bandit.arm_groups(1,2).return = 0.45;
            bandit.arm_groups(1,3).return = 0.4;
            bandit.arm_groups(1,4).return = 0.3;
            % Returns for group 2 are (0.5, 0.3, 0.2, 0.1)
            bandit.arm_groups(2,1).return = 0.5;
            bandit.arm_groups(2,2).return = 0.3;
            bandit.arm_groups(2,3).return = 0.2;
            bandit.arm_groups(2,4).return = 0.1;
            bandit.reset_arms(1);
            return
        end
        
        function [bandit] = clone_returns(a_returns)
            % Make a bernoulli bandit and set its arm returns to match those in
            % a_returns (a matrix of size group_count x arm_count)
            g_count = size(a_returns,1);
            a_count = size(a_returns,2);
            bandit = MultiArmBandit(g_count, a_count, 'bernoulli');
            for g=1:g_count,
                for a=1:a_count,
                    bandit.arm_groups(g,a).return = a_returns(g,a);
                end
            end
            bandit.reset_arms(1);
            return
        end     
    end
    
end





%%%%%%%%%%%%%%
% EYE BUFFER %
%%%%%%%%%%%%%%

