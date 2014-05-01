classdef NormalMaker < handle
    % NormalMaker makes normal bandits with various arm distributions.
    % 
    
    properties
        % group_count is the number of arm groups for this bandit
        group_count
        % arm_count is the number of arms per arm group for this bandit
        arm_count
        % return_type gives the return distribution: 'bernoulli' or 'normal'.
        return_type
    end
    
    methods
        function [self] = NormalMaker(group_count, arm_count)
            % Constructor for a normal bandit maker
            self.group_count = group_count;
            self.arm_count = arm_count;
            self.return_type = 'normal';
            return
        end
        
        function [bandit] = distro_1(self)
            % Make a normal bandit and set its arm returns. This is the
            % simplest return distribution; each arm's return is to a normal
            % distribution with mean randn() and unit variance.
            bandit = MultiArmBandit(...
                self.group_count, self.arm_count, 'normal');
            for g=1:bandit.group_count,
                group_returns = randn(1,self.arm_count);
                group_returns = sort(group_returns,'descend');
                for a=1:bandit.arm_count,
                   bandit.arm_groups(g,a).return = group_returns(a);
                   bandit.arm_groups(g,a).sigma = 1.0;
                end
            end
            bandit.reset_arms(1);
            return
        end
        
        function [bandit] = distro_2(self, top_m, easy_gap, hard_gap, hard_rate)
            % Make a normal bandit and set its arm returns. For this
            % distribution, "hard" arms occur at rate hard_rate. The return for
            % "easy" arms is at least easy_gap, and the return for hard arms is
            % at most hard_gap. The gap location is set by top_m.
            easy_count = self.group_count - round(self.group_count*hard_rate);
            bandit = MultiArmBandit(...
                self.group_count, self.arm_count, 'normal');
            for g=1:bandit.group_count,
                if (g > easy_count)
                    % Pick an arm return distribution such that the difference
                    % between its top_m and top_m+1 largest returns is less
                    % than hard_gap.
                    good_dist = 0;
                    while (good_dist == 0)
                        a_returns = randn(1,bandit.arm_count);
                        a_returns = sort(a_returns,'descend');
                        if (a_returns(top_m) - a_returns(top_m+1) < hard_gap)
                            good_dist = 1;
                        end
                    end
                    for a=1:bandit.arm_count,
                        bandit.arm_groups(g,a).return = a_returns(a);
                        bandit.arm_groups(g,a).sigma = 1.0;
                    end
                else
                    % Pick an arm return distribution such that the difference
                    % between its largest and second largest returns is more
                    % than easy_gap.
                    good_dist = 0;
                    while (good_dist == 0)
                        a_returns = randn(1,bandit.arm_count);
                        a_returns = sort(a_returns,'descend');
                        if (a_returns(top_m) - a_returns(top_m+1) > easy_gap)
                            good_dist = 1;
                        end
                    end
                    for a=1:bandit.arm_count,
                        bandit.arm_groups(g,a).return = a_returns(a);
                        bandit.arm_groups(g,a).sigma = 1.0;
                    end
                end     
            end
            bandit.reset_arms(1);
            return
        end
        
        function [bandit] = distro_3(self, top_m, min_gap, max_gap)
            % Make a normal bandit and set its arm returns. For this
            % distribution, the minimum gap between arm top_m and top_m+1 is
            % min_gap and the maximum gap is max_gap.
            bandit = MultiArmBandit(...
                self.group_count, self.arm_count, 'normal');
            for g=1:bandit.group_count,
                % Condition distribution on having min_gap < gap < max_gap.
                good_dist = 0;
                while (good_dist == 0)
                    a_returns = randn(1,bandit.arm_count);
                    a_returns = sort(a_returns,'descend');
                    if ((a_returns(top_m) - a_returns(top_m+1) > min_gap) &&...
                            (a_returns(top_m) - a_returns(top_m+1) < max_gap))
                        good_dist = 1;
                    end
                end
                for a=1:bandit.arm_count,
                    bandit.arm_groups(g,a).return = a_returns(a);
                    bandit.arm_groups(g,a).sigma = 1.0;
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

