function params = Parameters()
% Parameters  :  Encapsulate the default values of experiments
%
% Details of the parameters:
%   nb_iter                max number of iterations
%   conv_criterion         convergence criterion
%   damping                damping coefficient
%   prior_u                prior on the matrix U
%   prior_v                prior on the matrix V
%   prior_u_option         prior parameters for U
%   prior_v_option         prior parameters for V
%   beta                   temperature parameter
%   seed                   seed for reproducibility
%

       params.nb_iter = 200;
       params.conv_criterion = 1e-10;
       params.damping = 0.7;
       params.prior_u = 'Binary'; % can be in {'Gauss', 'Binary', 'Bernoulli-Gauss'}
       params.prior_v = 'Bernoulli-Gauss'; % can be in {'Gauss', 'Bernoulli-Gauss'}
       params.prior_u_option = struct('rho', 0.05); % sparsity rate for Bernoulli-Gauss       
       params.prior_v_option = struct('rho', 0.05); % sparsity rate for Bernoulli-Gauss
       params.beta = 1;
       params.seed = 12345;
end