% clear
% close all
% load('2024-01-10-10-58-55_deflated_sysID_long_resampled_0-05.mat')
% load('2023-12-12-15-03-55_inflated_sysID_long_resampled_0-05.mat')

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

% Test the merged data
% X_all = X_all_new;
% U_all = U_all_new;
len = length(X_all);

%% Calculate AB
% History size
n = 5;
% Window size
w = 1395;
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

% AB_display = AB1(:,:,t0:t0 + 100);

%% Test
% Z_sim = zeros(9, len);
% for j = t0:len
%     AB = AB1(:,:,j);
%     Z = [];
%     for i = 1:n
%         Z = [Z; X_all(:,j-i)];
%     end
%     U = U_all(:,j-1);
%     Z_sim(:, j) = AB * [Z; U];
% end
% 
% norm(Z_sim(:,t0:end) - X_all(:,t0:end))
% norm(Z_sim(:,513:517) - X_all(:,513:517))
% 
% figure;
% plot(Z_sim(1:3,t0:end)', 'r', 'LineWidth',2)
% hold on
% plot(X_all(1:3,t0:end)', 'b--', 'LineWidth',2)
% plot(U_all(1:3,t0:end)', 'k', 'LineWidth',2)
% title("drone xyz from time" + ' ' + t0)
% 
% figure;
% plot(Z_sim(7:9,t0:end)', 'LineWidth',2)
% hold on
% plot(X_all(7:9,t0:end)', '--', 'LineWidth',2)
% title("tip xyz from time" + ' ' + t0)

% figure;
% % sim_times = (t0:train_size)*.05;
% sim_times = (t0:len)*.05;
% plot(sim_times, Z_sim(1,t0:end)', 'b', 'LineWidth',2)
% hold on
% plot(sim_times, X_all(1,t0:end)', 'r--', 'LineWidth',2)
% plot(sim_times(1:end-1), U(1,:), 'k', 'LineWidth',2)
% xlabel("Time (s)")
% ylabel("x position (m)")
% title("drone x position from time" + ' ' + t0)
% legend(["sim" "real" "command"])
% 
% figure;
% plot(sim_times, Z_sim(7,t0:end)', 'b', 'LineWidth',2)
% hold on
% plot(sim_times, X_all(7,t0:end)', 'r--', 'LineWidth',2)
% title("tip x position from time" + ' ' + t0)
% xlabel("Time (s)")
% ylabel("x position (m)")
% legend(["sim" "real"])
% 
% figure;
% plot(sim_times, Z_sim(2,t0:end)', 'b', 'LineWidth',2)
% hold on
% plot(sim_times, X_all(2,t0:end)', 'r--', 'LineWidth',2)
% plot(sim_times(1:end-1), U(1,:), 'k', 'LineWidth',2)
% xlabel("Time (s)")
% ylabel("y position (m)")
% title("drone y position from time" + ' ' + t0)
% legend(["sim" "real" "command"])
% 
% figure;
% plot(sim_times, Z_sim(3,t0:end)', 'b', 'LineWidth',2)
% hold on
% plot(sim_times, X_all(3,t0:end)', 'r--', 'LineWidth',2)
% plot(sim_times(1:end-1), U(1,:), 'k', 'LineWidth',2)
% xlabel("Time (s)")
% ylabel("y position (m)")
% title("drone z position from time" + ' ' + t0)
% legend(["sim" "real" "command"])

%% Validation
% n = 10; w = 1190; t0 = n + w + 1;
% j = t0;
% X_plus = X_all(:,j-w:j-1);
% U = U_all(:,j-w-1:j-2);
% Z = [];
% for i = 1:n
%     Z = [Z; X_all(:,j-w-i:j-1-i)];
% end
% AB_display = X_plus / ([Z; U]);



