target = transpose([10, 10, 2]);
for i = 1:1400
    control_inputs = repmat(target, 1, 1400);
end
save('control_inputs_manual.mat', 'control_inputs');