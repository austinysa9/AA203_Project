% clear
% close all
% Load

%% Data
load('2024-05-24-11-50-21_gental_resampled_0-05.mat')
num_skips = 90;
train_size = 295;
local_position1 = local_position(1:num_skips:train_size,:);
local_setpoint1 = local_setpoint(1:num_skips:train_size,:);
tip_pose1 = tip_pose(1:num_skips:train_size,:);
vision_pose = vision_pose(1:num_skips:train_size,:);

load('2024-05-24-11-40-06_traj_2.5_resampled_0-05.mat')
local_position2 = local_position(1:num_skips:train_size,:);
local_setpoint2 = local_setpoint(1:num_skips:train_size,:);
tip_pose2 = tip_pose(1:num_skips:train_size,:);
vision_pose = vision_pose(1:num_skips:train_size,:);


% (drone position, drone angles, tip position)
X_all = [local_position1(:,[2,3,4,6,7,8]) tip_pose1(:,2:4); local_position2(:,[2,3,4,6,7,8]) tip_pose2(:,2:4)]';
U_all = [local_setpoint1(:, 2:4); local_setpoint2(:, 2:4)]';

%% Data
% Test the merged data
% X_all = X_all_new;
% U_all = U_all_new;
len = length(X_all);

%% Calculate AB
% History size
n = 1;
% Window size
w = 999;
% Predict (Totally need w + n data)
% At time step t0 = n + w + 1
% [x(n+1)...x(w+n)] = [A B] [x(n) ... x(w+n-1)]
%                             ..
%                            x(1) ... x(w)
%                            u(n) ... u(w+n-1)
% Size: 9 * w = [A B] * (9n + 3) * w
t0 = n + w + 1;
% Fixed amount of data m
% AB1 = zeros(9, 9 * n + 3, 1200);
AB5 = 0;

for j = t0:len + 1
    X_plus = X_all(:,j-w:j-1);
    U = U_all(:,j-w-1:j-2);
    Z = [];
    for i = 1:n
        Z = [Z; X_all(:,j-w-i:j-1-i)];
    end

    % AB1(:, :, j) = X_plus / ([Z; U]);
    AB5 = X_plus / ([Z; U]);
end