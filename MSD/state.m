clear; clc;

m = 1;  b = 0.5;  k = 2;

n_samples_desired = 10000;
base_dt   = 0.3;
k_perturb = 0.10;
noise_type = 'normal';    % 'normal' | 'uniform'
noise_ratio = 0.0;          % e.g. 0.05  →  5 % noise

switch noise_type
    case 'normal',  noise = randn(n_samples_desired,1);
    case 'uniform', noise = rand(n_samples_desired,1) - 0.5;
    otherwise,      error("noise_type must be 'normal' or 'uniform'.");
end
delta_t   = max(base_dt + k_perturb*noise, 0.05);
timestamps = cumsum(delta_t);
T_end      = timestamps(end);
n_samples  = numel(timestamps);

u_k      = randn(n_samples,1);
u_t      = [0; timestamps];          % pad t = 0
u_values = [u_k(1); u_k];            % pad first sample
u_interp = @(t) interp1(u_t, u_values, t, 'previous');  % safe everywhere

A = [0 1; -k/m -b/m];   B = [0; 1/m];   C = [1 0];
ode_func = @(t,x) A*x + B*u_interp(t);

opts = odeset('RelTol',1e-6,'AbsTol',1e-9,'MaxStep',0.01);
x0   = [0;0];

[t_int, x_full] = ode45(ode_func,[0 T_end],x0,opts);
x = interp1(t_int, x_full, timestamps, 'linear', 'extrap');   % n×2

y0      = x(:,1);                       % position
sigma_y = std(y0);
y_noisy = y0 + noise_ratio*sigma_y*randn(size(y0));

data = table( u_k, y_noisy, timestamps, [0; diff(timestamps)], x, ...
              'VariableNames',{'Input','Output','Time','Delta_t','TrueState'});

writetable(data,'MSD_linear_noiseless_k_010.csv');

figure;
plot(timestamps, y_noisy, '.-');
grid on;
xlabel('Time (s)'); ylabel('Position (m)');
title('Noisy System Output over Time');
