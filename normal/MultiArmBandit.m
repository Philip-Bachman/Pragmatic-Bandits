classdef MultiArmBandit < handle
    % MultiArmBandit implements both normal and bernoulli type bandits.
    % This class constructs multi armed bandit with gruoped arms that provides
    % either normally or bernoulli distributed returns. The required parameters
    % at instantiation are: the number of arm groups, the number of arms per
    % group, and the return type. 
    
    properties
        % arm_groups is an array of struct arrays, with each struct array
        % containing an arm group. Each arm group struct array in arm_groups
        % contains all of the arm structs belonging to the given arm group. An
        % arm struct contains the following:
        % Bernoulli arms:
        %   arm.return: rate of return for this arm
        %   arm.pulls: number of times this arm has been pulled
        %   arm.pull_seq: the sequence of returns to use for each pull, we use
        %                 this to permit "repeated" experiments.
        % Normal arms:
        %   arm.return: rate of return for this arm
        %   arm.sigma: standard deviation of return for this arm
        %   arm.pulls: number of times this arm has been pulled
        %   arm.pull_seq: the sequence of returns to use for each pull, we use
        %                 this to permit "repeated" experiments.
        %
        arm_groups
        % group_count is the number of arm groups for this bandit
        group_count
        % arm_count is the number of arms per arm group for this bandit
        arm_count
        % return_type gives the return distribution: 'bernoulli' or 'normal'.
        return_type
    end
    
    methods
        function [self] = MultiArmBandit(group_count, arm_count, return_type)
            % Constructor for multiarmed bandits
            if (~strcmpi(return_type,'bernoulli') && ...
                    ~strcmpi(return_type,'normal'))
                error('MultiArmBandit: return_type set incorrectly.');
            end
            % Capture parameters
            self.group_count = group_count;
            self.arm_count = arm_count;
            self.return_type = return_type;
            % Initialize arm_groups with enough groups/arms
            self.arm_groups = struct();
            for g_num=1:group_count,
                for a_num=1:arm_count,
                    if strcmpi(return_type,'bernoulli')
                        self.arm_groups(g_num,a_num).return = 0.5;
                        self.arm_groups(g_num,a_num).pulls = 0;
                    else
                        self.arm_groups(g_num,a_num).return = 1.0;
                        self.arm_groups(g_num,a_num).sigma = 0.1;
                        self.arm_groups(g_num,a_num).pulls = 0;
                    end
                end
            end
            % Initialize returns for each arm
            self.set_returns();
            return
        end
        
        function [ arm_groups ] = set_returns(self, return_opts)
            % Set the return properties for each arm in self.arm_groups.
            if ~exist('return_opts','var')
                return_opts = struct();
            end
            for g_num=1:self.group_count,
                for a_num=1:self.arm_count,
                    if strcmpi(self.return_type,'bernoulli')
                        r = rand();
                        p = rand(1,2500) < r;
                        self.arm_groups(g_num,a_num).return = r;
                        self.arm_groups(g_num,a_num).pull_seq = p;
                    else
                        r = randn();
                        s = 0.1;
                        p = (randn(1,2500) .* s) + r;
                        self.arm_groups(g_num,a_num).return = r;
                        self.arm_groups(g_num,a_num).sigma = s;
                        self.arm_groups(g_num,a_num).pull_seq = p;
                    end
                end
            end
            arm_groups = self.arm_groups;
            return
        end
        
        function [ result ] = pull_arm(self, group, arm)
            % Pull arm 'arm_num' in group 'group_num'
            try
                pulls = self.arm_groups(group,arm).pulls;
            catch err
                error('MultiArmBandit.pull_arm(): invalid group/arm num.');
            end
            result = self.arm_groups(group,arm).pull_seq(pulls + 1);
            self.arm_groups(group,arm).pulls = pulls + 1;
            return
        end
        
        function [ result ] = reset_arms(self, reset_returns)
            % Reset the pull count on all arms to 0. If reset_returns==1, then
            % generate a new pull sequence for each arm too.
            for g_num=1:self.group_count,
                for a_num=1:self.arm_count,
                    self.arm_groups(g_num,a_num).pulls = 0;
                end
            end
            if (reset_returns == 1)
                if strcmpi(self.return_type,'bernoulli')
                    for g=1:self.group_count,
                        for a=1:self.arm_count,
                            r = self.arm_groups(g,a).return;
                            p = rand(1,2500) < r;
                            self.arm_groups(g,a).pull_seq = p;
                        end
                    end
                else
                    for g=1:self.group_count,
                        for a=1:self.arm_count,
                            r = self.arm_groups(g,a).return;
                            s = self.arm_groups(g,a).sigma;
                            p = (randn(1,2500) .* s) + r;
                            self.arm_groups(g,a).pull_seq = p;
                        end
                    end                    
                end
            end 
            result = 1;
            return
        end
    end
    
end





%%%%%%%%%%%%%%
% EYE BUFFER %
%%%%%%%%%%%%%%

