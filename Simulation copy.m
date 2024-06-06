%% Load Models
load('Models_5.mat')

%% Load Data Manually
load('2024-01-10-10-58-55_deflated_sysID_long_resampled_0-05.mat')

%% Data
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

n = 5;
t0 = n + 1;

AB = [AB1, AB2, AB3, AB4, AB5];
d = ones(len, 1);
model = zeros(len, 1);

Z_sim = zeros(9, len, 5);
for j = t0:len
    % AB = AB2;
    Z = [];
    for i = 1:n
        Z = [Z; X_all(:,j-i)];
    end
    U = U_all(:,j-1);
    Z_sim1(:,j) = AB1* [Z; U];
    Z_sim2(:,j) = AB2* [Z; U];
    Z_sim3(:,j) = AB3* [Z; U];
    Z_sim4(:,j) = AB4* [Z; U];
    Z_sim5(:,j) = AB5* [Z; U];

    for k = 1:5
        Z_sim(:,j,k) = AB(:,48*(k-1) + 1:48*k)* [Z; U];
        if norm(Z_sim(:, j, k) - X_all(:, j)) < d(j, 1)
            d(j, 1) = norm(Z_sim(:, j, k) - X_all(:, j));
            model(j, 1) = k;
        end
    end
end

% Whole Difference
norm(Z_sim1(:,t0:end) - X_all(:,t0:end))
norm(Z_sim2(:,t0:end) - X_all(:,t0:end))
norm(Z_sim3(:,t0:end) - X_all(:,t0:end))
norm(Z_sim4(:,t0:end) - X_all(:,t0:end))
norm(Z_sim5(:,t0:end) - X_all(:,t0:end))

% Column Difference
norm_diff1 = vecnorm(Z_sim1(:,t0:end) - X_all(:,t0:end));
norm_diff2 = vecnorm(Z_sim2(:,t0:end) - X_all(:,t0:end));
norm_diff3 = vecnorm(Z_sim3(:,t0:end) - X_all(:,t0:end));
norm_diff4 = vecnorm(Z_sim4(:,t0:end) - X_all(:,t0:end));
norm_diff5 = vecnorm(Z_sim5(:,t0:end) - X_all(:,t0:end));

% Find the maximum norm difference
max_norm_diff1 = max(norm_diff1)
max_norm_diff2 = max(norm_diff2)
max_norm_diff3 = max(norm_diff3)
max_norm_diff4 = max(norm_diff4)
max_norm_diff5 = max(norm_diff5)


figure;
plot(Z_sim1(1,t0:end)', 'r', 'LineWidth',1)
hold on
plot(X_all(1,t0:end)', 'b--', 'LineWidth',2)
plot(Z_sim2(1,t0:end)', 'r', 'LineWidth',1)
plot(Z_sim3(1,t0:end)', 'r', 'LineWidth',1)
plot(Z_sim4(1,t0:end)', 'r', 'LineWidth',1)
plot(Z_sim5(1,t0:end)', 'r', 'LineWidth',1)

%plot(U_all(1,t0:end)', 'k', 'LineWidth',2)
legend('Simulation','Experiment')
title("drone x position")

figure;
plot(Z_sim1(7,t0:end)', 'r', 'LineWidth',1)
hold on
plot(X_all(7,t0:end)', 'b--', 'LineWidth',2)
plot(Z_sim2(7,t0:end)', 'r', 'LineWidth',1)
plot(Z_sim3(7,t0:end)', 'r', 'LineWidth',1)
plot(Z_sim4(7,t0:end)', 'r', 'LineWidth',1)
plot(Z_sim5(7,t0:end)', 'r', 'LineWidth',1)

%plot(U_all(1,t0:end)', 'k', 'LineWidth',2)
legend('Simulation','Experiment')
title("Tip X Position")

figure;
plot(Z_sim1(7,t0:end)', 'LineWidth',2)
hold on
plot(Z_sim2(7,t0:end)', 'r', 'LineWidth',2)
plot(Z_sim3(7,t0:end)', 'r', 'LineWidth',2)
plot(Z_sim4(7,t0:end)', 'r', 'LineWidth',2)
plot(Z_sim5(7,t0:end)', 'r', 'LineWidth',2)
plot(X_all(7,t0:end)', '--', 'LineWidth',2)
title("tip x position")

% figure;
% plot(Z_sim1(1:3,t0:end)', 'r', 'LineWidth',2)
% hold on
% plot(X_all(1:3,t0:end)', 'b--', 'LineWidth',2)
% plot(U_all(1:3,t0:end)', 'k', 'LineWidth',2)
% title("drone xyz from time" + ' ' + t0)
% 
% figure;
% plot(Z_sim1(7:9,t0:end)', 'LineWidth',2)
% hold on
% plot(X_all(7:9,t0:end)', '--', 'LineWidth',2)
% title("tip xyz from time" + ' ' + t0)
% 
% figure;
% plot(Z_sim1(1:3,t0:end)', 'r', 'LineWidth',1)
% hold on
% plot(Z_sim2(1:3,t0:end)', 'b--', 'LineWidth',1)
% plot(Z_sim3(1:3,t0:end)', 'k', 'LineWidth',1)
% plot(Z_sim4(1:3,t0:end)', 'y', 'LineWidth',1)
% plot(Z_sim5(1:3,t0:end)', 'g', 'LineWidth',1)
% title("drone xyz from time" + ' ' + t0)
% 
% figure;
% plot(Z_sim1(7:9,t0:end)', 'r', 'LineWidth',1)
% hold on
% plot(X_all(7:9,t0:end)', 'b--', 'LineWidth',1)
% plot(Z_sim2(7:9,t0:end)', 'b', 'LineWidth',1)
% plot(Z_sim3(7:9,t0:end)', 'k', 'LineWidth',1)
% plot(Z_sim4(7:9,t0:end)', 'y', 'LineWidth',1)
% plot(Z_sim5(7:9,t0:end)', 'g', 'LineWidth',1)
% title("drone xyz from time" + ' ' + t0)