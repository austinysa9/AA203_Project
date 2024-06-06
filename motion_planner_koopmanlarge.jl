using IterativeLQR 
using LinearAlgebra
using Plots
using MAT
using CSV
using DataFrames
using NPZ

# Load the .mat file
# mat_file = matopen("example_dynamics.mat")
# mat_file = matopen("inflated_sysid_realistic_1_dynamics.mat")
# mat_file = matopen("deflated_sysID_nice-resample_dt0-05.mat")
mat_file = matopen("koopman_large_ABfull.mat")
dt = 0.05

A_full = read(mat_file, "A_full")
B_full = read(mat_file, "B_full")
close(mat_file)

# horizon 
T = 25

# koopman 
state_size = 9
history_size = 5
num_state = state_size*history_size
num_action = 3 

coefficients = [
    -0.12484799100377, 2.73287714025537, 0.236483012926316, 1.40823939569275, -0.12974448358976, 0.612050927357037,
    -9.40391400001946, -0.226686894977195, -0.369104958313975, -2.28734014786361, 7.54147545891863, 0.464310162855565,
    1.74896015925701, -0.669219493779387, 0.0121658261410173, -0.167363162508408, -0.667621671734472, 8.7874748557572,
    -14.925230312804, 10.7359287961859, -2.70235776286358, -1.92370949730096, 1.94926926714097, 1.37394980747646,
    4.27143480987716, -2.91195842436634, -2.78714967726693, 1.18414944043454, -1.13011559329765, 23.9948928355406,
    -9.30929727141083, -0.650845560045152, -14.1701877025959, 1.6765940777675, 1.04914001012749, 2.31268122232003,
    5.50626102276125, -2.09483873110315, -4.20105614554877, 0.765566369922597, 11.505753302943, 7.8409896285926,
    -1.93036563270671, -4.05863437242042, 5.1933997719092, -3.84106886089721, 0.230492670897789, -0.333548622191016,
    -5.22508587898739, 7.92359488096077, 5.81887021279646, -9.22663285744954, -6.99670819508162, 1.67235798799438,
    1.29386362949991, -4.77907567881161, -4.61722343839485, 2.00199056831667, 3.27586909644781, -2.68457762512331,
    -0.567736522634023, 2.52926925983532, 1.89101611599747, 5.19672646167121, -0.55607918420966, -6.60090833561978,
    -12.3221739485307, 2.21694154066915, 0.110714662363752, -5.27019975453913, 9.49000552802504, -4.64113258701139,
    -1.14702241740954
]


function koopman_discrete(x, u)
   # xnext = A_full * x + B_full * [u[1]; 0; 1.5] 
   xnext = A_full * x + B_full * u 
   terms = [
      1, x[3], x[5], x[9], x[14], x[18], x[21], x[22], x[23], x[27], x[30], x[32], x[36], x[39], x[40], x[41], x[45],
      x[3]*x[9], x[3]*x[18], x[3]*x[27], x[3]*x[30], x[3]*x[36], x[3]*x[41], x[5]*x[18], x[5]*x[23], x[5]*x[36], x[5]*x[41], x[5]*x[45], x[9]*x[14],
      x[9]*x[18], x[9]*x[21], x[9]*x[22], x[9]*x[27], x[9]*x[39], x[9]*x[41], x[9]*x[45], x[14]*x[36], x[14]*x[41], x[14]*x[45], x[18]*x[22], x[18]*x[30],
      x[18]*x[36], x[18]*x[41], x[18]*x[45], x[21]*x[39], x[21]*x[41], x[22]*x[39], x[22]*x[45], x[23]*x[36], x[23]*x[41], x[23]*x[45], x[27]*x[36],
      x[27]*x[39], x[27]*x[41], x[27]*x[45], x[30]*x[39], x[30]*x[45], x[32]*x[36], x[32]*x[41], x[32]*x[45], x[36]*x[41], x[36]*x[45], x[39]*x[41],
      x[39]*x[45], x[5]^2, x[9]^2, x[18]^2, x[21]^2, x[22]^2, x[23]^2, x[27]^2, x[41]^2, x[45]^2
  ]
   xnext[9] = dot(coefficients, terms)
   return xnext
end

# model
koopman = Dynamics(koopman_discrete, num_state, num_action)
model = [koopman for t = 1:T-1] 

# x = (drone pos, drone angles, tip pos)
# z = (x_t, x_t-1, ... x_t-9)
function gen_z_from_drone(drone_pos)
   z = zeros(num_state)

   # drone position
   z[1:9:end] .= drone_pos[1]
   z[2:9:end] .= drone_pos[2]
   z[3:9:end] .= drone_pos[3]
   
   # tip angles
   z[7:9:end] .= drone_pos[1] 
   z[8:9:end] .= drone_pos[2]
   z[9:9:end] .= drone_pos[3] - .9

   return z
end

# initialization
x1 = gen_z_from_drone([0, 0, 1.5])

# desired trajectory
xref = [gen_z_from_drone([1, 0, 1.5]) for i=0:T-1]
uref = [[1, 0, 1.5] for i=10 .+ (1:T-1)]

h = plot([x[7] for x in xref], [x[8] for x in xref], label="ref")
plot!([u[1] for u in uref], [u[2] for u in uref], label="control")
display(h)

# adjust for history
for i=2:T
   for j=1:(history_size-1)      
      xref[i][9j .+ (1:9)] = xref[max(i-j, 1)][1:9]
   end
end

# initial rollout
ū = deepcopy(uref)
x̄ = rollout(model, x1, ū)
x̄array = permutedims(hcat(x̄...))'

h = plot([x[7] for x in xref], [x[9] for x in xref], linewidth = 2, label="desired tip position", aspect_ratio=:equal)
plot!([x[1] for x in uref], [x[3] for x in uref], linewidth = 2, aspect_ratio=:equal, label="drone command")
plot!(x̄array[1,:], x̄array[3,:], aspect_ratio=:equal, linewidth = 2, label="drone position")
plot!(x̄array[7,:], x̄array[9,:], linewidth = 2, label="tip position", legend=:bottomleft, aspect_ratio=:equal)
xlabel!("x (m)")
ylabel!("y (m)")
title!("Initial Rollout")
display(h)

## objective
# qt = [i < 10 ? ones(num_state) : ones(num_state) for i=1:T] # more weight on position
qt = ones(num_state)
qt[7:9] .= 5.0 # more weight on tip position
rt = ones(num_action)

# objective  
ots = [(x, u) -> transpose(x - xref[t]) * Diagonal(qt) * (x - xref[t]) +
      transpose(u - uref[t]) * Diagonal(rt) * (u - uref[t]) for t = 1:T-1]
oT = (x, u) -> transpose(x - xref[T]) * Diagonal(qt) * (x - xref[T])

cts = [Cost(ot, num_state, num_action) for ot in ots]
cT = Cost(oT, num_state, 0)
objective = [cts..., cT]

# constraints
u_min, u_max, x_min, x_max = -3.0, 3.0, -2.0, 2.0
function limits(x, u) 
   [  -u .+ u_min; 
      u .- u_max;
      -x[1:3] .+ x_min;
      x[1:3] .- x_max;
      x[1] - 0.7;
   ]
end 
function goal(x, u)
   [  #-x[9] + 0.7;
      -(x[18:9:45] - x[9:9:36]);
      -(x[16:9:43] - x[7:9:34]);
      # x[7] - xref[T][7]]
      x[7] - 1.1]
end
cont = Constraint(limits, num_state, num_action, 
    indices_inequality=collect(1:length(limits(zeros(num_state), zeros(num_action))))) # Assume all are inequality constraints
# cont = IterativeLQR.Constraint()
conT = IterativeLQR.Constraint(goal, num_state, 9)
# conT = Constraint()
constraints = [[cont for t = 1:T]...]
constraints[end] = conT

# solver
solver = Solver(model, objective, constraints, 
      options=IterativeLQR.Options(
      verbose=true,
      line_search=:armijo,
      min_step_size=1.0e-5,
      objective_tolerance=1.0e-4,
      constraint_tolerance=1.0e-2,
      lagrangian_gradient_tolerance=1.0e-4,
      max_iterations=500,#250,
      max_dual_updates=6,#15,
      initial_constraint_penalty=1.0,
      scaling_penalty=10.0))
initialize_controls!(solver, ū)
initialize_states!(solver, x̄)

# solve
solve!(solver)

# solution
x_sol, u_sol = get_trajectory(solver)

# extract policy
K = vcat(solver.policy.K...)
k = vcat(solver.policy.k...)'
Xnom = permutedims(hcat(x_sol...))'
Unom = permutedims(hcat(u_sol...))'

# plot
h=plot()
for i in 1:3
   for j in 1:45
      plot!([x[i,j] for x in solver.policy.K], legend=false)
   end
end
display(h)

plot(k)
display(plot(Unom', title="controls"))
display(plot(Xnom[1:3, :]', title="drone pos")) # plot drone position
plot(Xnom[7:9,:]', title="tip pos") # plot tip position
plot!([x[7] for x in xref], linewidth = 2, label="desired tip position")

h = plot([x[7] for x in xref], [x[9] for x in xref], label="desired tip position", linewidth=2)
plot!(Unom[1,:], Unom[3,:], aspect_ratio=:equal, label="drone position command", linewidth=2)
plot!(Xnom[1,:], Xnom[3,:], aspect_ratio=:equal, label="drone position", linewidth=2)
plot!(Xnom[7,:], Xnom[9,:], label="tip position", linewidth=2)
display(h)

h = plot([x[7] for x in xref], label="desx")
plot!(Xnom[7,:], label="tipx")
display(h)

h = plot([x[9] for x in xref], label="desz")
plot!(Xnom[9,:], label="tipz")
display(h)

println("done")

# # save reference trajectory
# motion_plan = zeros(14, T)
# motion_plan[1, :] = 0:dt:(T-1)*dt
# motion_plan[2:4, 1] = x1[1:3]
# motion_plan[2:4, 2:end] .= Unom[1:3, :]
# motion_plan[8, :] .= 1

# npzwrite("2024_05_13_kick_uxyz_gxz.npy", motion_plan)

# save lqr controller
npzwrite("2024_05_22_kick_uxyz_gx_dronexlim_tunevinelength.npz", Dict("u_nom" => Unom, "x_nom" => Xnom, "K" =>  solver.policy.K[1]))
