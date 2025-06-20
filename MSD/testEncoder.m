clear all
clc 
close all

% dataTable = readtable('mass_string_damper_k_0_2.csv')
dataTable = readtable('mass_string_damper_k_0_5.csv')
DT = dataTable.Delta_t; 
DT = [DT(2:end); 0]; % shift DT one sample backwards
T = dataTable.Time;
x1 = dataTable.TrueState_1;
x2 = dataTable.TrueState_2;
u = dataTable.Input;
y = dataTable.Output;


N = length(u);
n = 5;  % number of past inputs, outputs, DT in encoder
K = zeros(N,n*3); % big matrix that will contain all the encoder inputs

for ii=1:n
    K(:,ii) = [zeros(ii-1,1); u(1:end-ii+1)];
    K(:,ii+n) = [zeros(ii-1,1); y(1:end-ii+1)];
    K(:,ii+2*n) = [zeros(ii-1,1); DT(1:end-ii+1)];
end
KEst = K(n+1:end-1,:);
xEst = [x1(n+2:end,:) x2(n+2:end,:)];

% set up simple feedforward ANN, 2 layers with [20, 10] neurons in each layer
net = feedforwardnet([20 10 5],'trainlm');
net.trainParam.max_fail = 100;

% train encoder with DT
netDT = net;
netDT = train(netDT,KEst.',xEst.');

% train encoder without DT
netSimple = net;
netSimple = train(netSimple,KEst(:,1:2*n).',xEst.');

% compare results - note I didn't follow a strict train, validation, test
% split here.
xEstDT = sim(netDT,KEst.'); xEstDT = xEstDT';
xEstSimple = sim(netSimple,KEst(:,1:2*n).'); xEstSimple = xEstSimple';

% plot
figure; tiledlayout(1,2);
nexttile; hold on; plot(xEst(:,1)); plot(xEst(:,1)-xEstSimple(:,1)); plot(xEst(:,1)-xEstDT(:,1));
nexttile; hold on; plot(xEst(:,2)); plot(xEst(:,2)-xEstSimple(:,2)); plot(xEst(:,2)-xEstDT(:,2));
legend('state','err. simple','err. DT')

disp('RMSe without DT:')
disp(['x1: ' num2str(rms(xEst(:,1)-xEstSimple(:,1)))])
disp(['x2: ' num2str(rms(xEst(:,2)-xEstSimple(:,2)))])
disp('RMSe with DT:')
disp(['x1: ' num2str(rms(xEst(:,1)-xEstDT(:,1)))])
disp(['x2: ' num2str(rms(xEst(:,2)-xEstDT(:,2)))])
