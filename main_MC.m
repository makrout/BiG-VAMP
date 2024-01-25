close all;
clear all;
clc;

addpath('functions/');
addpath('BiG-VAMP-MC/');

% size of the matrices U (nxr) and V^T (mxr)
n=1000;
m=1000;
r=10;

% SNR level in dB
SNR_dB = 20;

% parameter object contaning the experiment's parameters
params = Parameters();

% fix the seed if needed
if params.seed
    rng(params.seed);
end

fprintf(1,'Generate U and V with n=%d, m=%d, and r=%d\n',m,n,r);

% set groundtruth value of U according to its prior
switch     params.prior_u
  case    {'Gauss'}  
    U = randn(n,r);
  case    {'Binary'} 
    U = 2*randi(2,n,r)-3;
  case    {'Bernoulli-Gauss'}
    sparsity = params.prior_u_option.rho;
    U = randn(n,r).*(rand(n,r)<sparsity);
  otherwise          
    error('U: unknown prior');
end
    
% set groundtruth value of V according to its prior
switch     params.prior_v
  case    {'Gauss'}  
    V = randn(m,r);        
  case    {'Bernoulli-Gauss'}
    sparsity = params.prior_v_option.rho;
    V = randn(m,r).*(rand(m,r)<sparsity);    
  otherwise          
    error('V: unknown prior')
end

% noiseless signal variance
var_UV = sum((U*V').^2, 'all')/prod(size(U*V'));

% variance of the gaussian noise to meed the predefined SNR level
var_w = var_UV * 10^(-SNR_dB/10);

% the noisy signal before the selection
Z = U*V'+ sqrt(var_w)*randn(n,m);

% the random selection
S = rand(size(Z)) < params.selection_percentage;

% Apply the random selection for matrix completion
Y = Z .* S;

% Run BiG-VAMP for matrix completion
fprintf(1,'Running BiG-VAMP for matrix completion \n');
[ u_est,v_est ]  = BiGVAMP_MC(Y, S, var_w, r, params);

% nrmse on the product U*V'
true_uv = U*V';
est_uv = u_est*v_est';
nrmse = sqrt(mean((true_uv - est_uv).^2/mean((true_uv).^2), 'all'));
fprintf(1,'NRMSE = %f \n', nrmse);


% Stem the recovered signal vs the true one
perm = randperm(m*n);
figure(1);
stem(true_uv(perm(1:1000)), 'b');
hold on
stem(est_uv(perm(1:1000)), '--r');
legend('true', 'estimated');
title(['BiG-VAMP for matrix completion: ', 'SNR=', num2str(SNR_dB), 'dB', ...
       ', n=', num2str(n), ', m=', num2str(m), ', r=', num2str(r)]);