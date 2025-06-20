% Parameters
m = 1;          % Mass (kg)
b = 0.5;        % Damping (N·s/m)
k = 2;          % Linear spring constant (N/m)
k_nl = 5;       % Nonlinear spring coefficient

n_samples = 20000;
base_dt = 0.3;
T_end = base_dt * n_samples;
k_perturb = 0.10; % series
noise_type = 'normal';  % or 'uniform'

% Generate delta_t
if strcmp(noise_type, 'normal')
    noise = randn(n_samples, 1);
elseif strcmp(noise_type, 'uniform')
    noise = rand(n_samples, 1) - 0.5;
else
    error('Invalid noise type.');
end

delta_t = base_dt + k_perturb * noise;
delta_t = max(delta_t, 0.01);  % prevent nonpositive
timestamps = cumsum(delta_t);
timestamps = timestamps(timestamps <= T_end);
n_samples = length(timestamps);

% Input signal u(t) as Gaussian white noise
u_k = randn(n_samples, 1);
u_interp = @(t) interp1(timestamps, u_k, t, 'previous', 'extrap');

% Nonlinear mass-spring-damper dynamics
ode_func = @(t, x) [x(2);
                    (-k*x(1) - k_nl*x(1)^3 - b*x(2) + u_interp(t)) / m];

% Solver settings
x0 = [0; 0];
options = odeset('RelTol', 1e-6, 'AbsTol', 1e-9);
[t, x] = ode45(ode_func, timestamps, x0, options);

% Output y(t)
y = x(:,1);

% Optional: create table and save
delta_t_out = [0; diff(t)];
data = table(u_interp(t), y, t, delta_t_out, x(:,1), x(:,2), ...
    'VariableNames', {'Input', 'Output', 'Time', 'Delta_t', 'x1', 'x2'});
writetable(data, 'MSD_nonlinear_noiseless_k_010.csv');

% Plot result
figure;
plot(t, y, 'b-');
title('Nonlinear Mass-Spring-Damper Output');
xlabel('Time (s)');
ylabel('Position y(t)');
grid on;
