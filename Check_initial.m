%% Load data
%                                                             Test:
% load('2024-02-15-14-16-26_resampled_0-05.mat') 29.125 in -- 0.8051 m
% --0.085725 m == 0.0842 m
% load('2024-02-15-14-42-06_resampled_0-05.mat') 40.125 in -- 1.0705 m

% load('2024-02-15-15-04-46_resampled_0-05.mat') 36.25 in -- 0.9723 m
% --0.098425 m == 0.0982 m
% load('2024-02-15-15-18-35_resampled_0-05.mat') 32.5 in -- 0.8893 m
% -- 0.09525 m == 0.083 m
% load('2024-02-15-15-30-31_resampled_0-05.mat') 24.625 in -- 0.6931 m
% --0.1143 m == 0.112 m

% Record      Test       Diff from Record    Diff from Test
% 40.125 in   1.0705 m
% 36.25 in    0.9723 m      0.098425 m         0.0982 m
% 32.5 in     0.8893 m      0.09525 m          0.083 m
% 29.125 in   0.8051 m      0.085725 m         0.0842 m
% 24.625 in   0.6931 m      0.1143 m           0.112 m


num_skips = 1;
train_size = 1400;
local_position = local_position(1:num_skips:train_size,:);
local_setpoint = local_setpoint(1:num_skips:train_size,:);
tip_pose = tip_pose(1:num_skips:train_size,:);
vision_pose = vision_pose(1:num_skips:train_size,:);

% (drone position, drone angles, tip position)
X_all = [local_position(:,[2,3,4,6,7,8]) tip_pose(:,2:4)]';
U_all = local_setpoint(:, 2:4)';
len = length(X_all);

%% Check the initial states to get the vine length
threshold = 1e-2;
x = 10;
y = 10;
z = 10;
idx = 0;
vine = 0;
for i=1:99
    if abs(X_all(1, i)) < threshold && abs(X_all(2, i)) < threshold && abs(X_all(3, i) - 1.5) < threshold
        xx = abs(X_all(1, i));
        yy = abs(X_all(2, i));
        zz = abs(X_all(3, i) - 1.5);
        if xx < x || yy < y || zz < z
            x = xx;
            y = yy;
            z = zz;
            idx = i;
            vine = X_all(3, i) - X_all(9, i);
        end

    end
end
