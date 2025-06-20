clear; clc; close all;

csv_path = 'MSD_linear_noiseless_k_040.csv';   
max_step = 0.01;                               % ODE MaxStep (s)
N_SHOW   = 10;                                 

TBL = readtable(csv_path,'VariableNamingRule','preserve');
t_s = TBL.Time  (1:N_SHOW);    % sample times
u_s = TBL.Input (1:N_SHOW);    % input samples
y_s = TBL.Output(1:N_SHOW);    % noisy output samples

m = 1;  b = 0.5;  k = 2;
A = [0 1; -k/m   -b/m];
B = [0; 1/m];

u_t = [0; t_s];              
u_v = [u_s(1); u_s];
u_interp = @(t) interp1(u_t, u_v, t, 'previous');
odefun = @(t,x) A*x + B*u_interp(t);
opts   = odeset('RelTol',1e-6,'AbsTol',1e-9,'MaxStep',max_step);
[t_int, x_full] = ode45(odefun, [0 t_s(end)], [0;0], opts);

figure('Color','w');
tiledlayout(2,1,'TileSpacing','compact');

% --- Time-Input 
nexttile;

t_plot_zoh = zeros(1, 2*N_SHOW);
u_plot_zoh = zeros(1, 2*N_SHOW);
if t_s(1) > 0
    t_plot_zoh(1) = 0;
    u_plot_zoh(1) = u_s(1);
    
    t_plot_zoh(2) = t_s(1);
    u_plot_zoh(2) = u_s(1);
    
    start_idx_t = 3;
    start_idx_u = 3;
else
    t_plot_zoh(1) = t_s(1);
    u_plot_zoh(1) = u_s(1);
    start_idx_t = 2;
    start_idx_u = 2;
end

% Loop through the rest of the sample points
for i = 1:N_SHOW-1
    % The value u_s(i) holds from t_s(i) until t_s(i+1)
    t_plot_zoh(start_idx_t) = t_s(i);
    u_plot_zoh(start_idx_u) = u_s(i);
    
    t_plot_zoh(start_idx_t + 1) = t_s(i+1);
    u_plot_zoh(start_idx_u + 1) = u_s(i); % This is the step
    
    start_idx_t = start_idx_t + 2;
    start_idx_u = start_idx_u + 2;
end

% For the last point, u_s(N_SHOW) holds from t_s(N_SHOW) till t_s(end)
t_plot_zoh(start_idx_t) = t_s(N_SHOW);
u_plot_zoh(start_idx_u) = u_s(N_SHOW);


% Remove any trailing zeros if the initial handling led to extra space
t_plot_zoh = t_plot_zoh(t_plot_zoh~=0 | u_plot_zoh~=0); % This is a rough way, better is to manage index count
u_plot_zoh = u_plot_zoh(1:length(t_plot_zoh));


% Clear the previously constructed, possibly problematic vectors
clear t_plot_zoh u_plot_zoh;

% Prepend (0, u_s(1)) if your first sample is not at t=0
if t_s(1) > 0
    t_zoh_nodes = [0; t_s];
    u_zoh_values = [u_s(1); u_s];
else
    t_zoh_nodes = t_s;
    u_zoh_values = u_s;
end

% Create the stepped plot points
t_plot_zoh = zeros(1, 2*length(t_zoh_nodes) - 1);
u_plot_zoh = zeros(1, 2*length(u_zoh_values) - 1);

for i = 1:length(t_zoh_nodes) - 1
    t_plot_zoh(2*i-1) = t_zoh_nodes(i);
    u_plot_zoh(2*i-1) = u_zoh_values(i);
    
    t_plot_zoh(2*i) = t_zoh_nodes(i+1);
    u_plot_zoh(2*i) = u_zoh_values(i);
end
% Add the very last point
t_plot_zoh(end) = t_zoh_nodes(end);
u_plot_zoh(end) = u_zoh_values(end);


plot(t_plot_zoh, u_plot_zoh, 'b-', 'LineWidth', 1.5, 'DisplayName','ZOH input'); hold on;
% Plot
plot(t_s, u_s, 'ob', 'MarkerSize',6, 'LineWidth',2.2, ...
     'DisplayName','sampled u_k');

grid on;
ylabel('u_k','FontWeight','bold');
title(sprintf('Time–Input (ZOH, first %d samples)', N_SHOW), ...
      'FontWeight','bold');
xlabel('time(s)','FontWeight','bold'); 
legend('Location','best','Box','off'); 
%Time-Output
nexttile;
% ODE trajectory
plot(t_int, x_full(:,1), 'Color',[0.35 0.35 0.35], ...
     'LineWidth',2.5, 'DisplayName','ODE trajectory'); hold on;
% sampling
plot(t_s, y_s, 'or', 'MarkerSize',6, 'LineWidth',2.2, ...
     'DisplayName','sampled y_k');
grid on;
xlabel('time(s)','FontWeight','bold');
ylabel('y_k','FontWeight','bold');
title(sprintf('Time–Output (first %d samples, MaxStep = %.3g s)', ...
      N_SHOW, max_step), 'FontWeight','bold');
legend('Location','best','Box','off');