num_skips = 1;
train_size = 340;
local_position = local_position(1:num_skips:train_size,:);
local_setpoint = local_setpoint(1:num_skips:train_size,:);
tip_pose = tip_pose(1:num_skips:train_size,:);
vision_pose = vision_pose(1:num_skips:train_size,:);

% (drone position, drone angles, tip position)
X_all = [local_position(:,[2,3,4,6,7,8]) tip_pose(:,2:4)]';
U_all = local_setpoint(:, 2:4)';
len = length(X_all);

figure;
plot(X_all(1,1:end)', 'r', 'LineWidth',2)
hold on
plot(X_all(7,1:end)', 'b-', 'LineWidth',0.5)
plot(X_all(9,1:end)', 'b-', 'LineWidth',0.5)
plot(U_all(1,1:end)', 'k', 'LineWidth',2)
