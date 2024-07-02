clear all;
close all;
clc;

% add the path of the bigamp package (from the GAMP Matlab package)
path(path, './bigamp-package/BiGAMP');
path(path, './bigamp-package/main');
% add the path of the BiG-VAMP algorithm
path(path, './functions/');
path(path, './BiG-VAMP-MC/');

%% Set the experiment parameters

% set BiGVAMP's parameters
params = Parameters();
params.damping = 0.8;
params.selection_percentage = 0.25;

% set the number of monte carlo
N_MC = 100;

% set the problem size
M = 1500;
L = 500;
N = 30; % the rank

% set the signal-to-noise ratio
SNR = 10;

fprintf(1,'Running BiG-AMP and BiG-VAMP with:\nA binary {-1, 1} and X Gaussian(0,1)\n\n');

%% run the benchmark for N_MC monte carlo
nrmses_bigamp = [];
nrmses_bigvamp = [];
ts_BiGAMP = [];
ts_BiGVAMP = [];

for n_mn = 1:N_MC

    n_mn
    %% Generate the Unknown Matrices to recover

    %Generate the two matrix factors, recalling the data model Z = AX
    % A is a binary {-1, 1} matrix
    A = 2*randi(2,M,N)-3;
    % X is Gaussian(0,1)
    X = randn(N,L);
    
    % Noise free signal
    Z = A*X;
    
    %Define the error function for computing normalized mean square error.
    %BiG-AMP will use this function to compute NMSE for each iteration
    error_function = @(qval) 20*log10(norm(qval - Z,'fro') / norm(Z,'fro'));
    
    %% AWGN Noise    
    %Determine the noise variance that is consistent with this SNR
    nuw = norm(reshape(Z,[],1))^2/M/L*10^(-SNR/10);
    
    %Generate noisy data
    Y = Z + sqrt(nuw)*randn(size(Z));    
    
    %% Observe a fraction of the noisy matrix entries
    
    %For matrix completion, we observe only a fraction of the entries of Y. We
    %denote this fraction as p1
    p1 = params.selection_percentage;    
    
    %Choose a fraction p1 of the entries to keep. Omega is an MxL matrix of
    %logicals that will store these locations
    omega = false(M,L);
    ind = randperm(M*L);
    omega(ind(1:ceil(p1*M*L))) = true;
    
    %Set the unobserved entries of Y to zero
    Y(~omega) = 0;    
    
    %% Define options for BiG-AMP
    
    %Set options
    opt = BiGAMPOpt; %initialize the options object with defaults
    
    %Use sparse mode for low sampling rates
    if p1 <= 0.2
        opt.sparseMode = 1;
    end
    
    %Provide BiG-AMP the error function for plotting NMSE
    opt.error_function = error_function;
    
    %Specify the problem setup for BiG-AMP, including the matrix dimensions and
    %sampling locations. Notice that the rank N can be learned by the EM code
    %and does not need to be provided in that case. We set it here for use by
    %the low-level codes which assume a known rank
    problem = BiGAMPProblem();
    problem.M = M;
    problem.N = N;
    problem.L = L;
    [problem.rowLocations,problem.columnLocations] = find(omega);
    
    
    %% Specify Prior objects for BiG-AMP
    
    %Note: The user does not need to build these objects when using EM-BiG-AMP,
    %as seen below.
    
    %First, we will run BiG-AMP with knowledge of the true distributions. To do
    %this, we create objects that represent the priors and assumed log
    %likelihood
    
    %Prior distribution on X is Gaussian. The arguments are the mean and variance.
    %Notice that we can use a scalar estimator that will be applied to each
    %component of the matrix.
    gX = AwgnEstimIn(0, 1);
    % Prior on A is binary {-1,1} with p(-1)=p(1)=0.5
    A0 = [-1 1]';
    pA0 = 0.5*[1 1]';
    
    gA_array = cell(M*N,1);
    for iii=1:M*N
       gA_array{iii} = DisScaEstim(A0, pA0);
    end
    gA = EstimInConcat(gA_array, ones(M*N,1)');
    
    
    %Log likelihood is Gaussian, i.e. we are considering AWGN
    if opt.sparseMode
        %In sparse mode, only the observed entries are stored
        gOut = AwgnEstimOut(reshape(Y(omega),1,[]), nuw);
    else
        gOut = AwgnEstimOut(Y, nuw);
    end
    
    %% Run BiG-AMP
    
    tstart = tic;
    [estFin,~,estHist] = BiGAMP_matrix(gX, gA, gOut, problem, opt);
    tBiGAMP = toc(tstart);
    ts_BiGAMP = [ts_BiGAMP tBiGAMP];

    Ahat = estFin.Ahat;
    Xhat = estFin.xhat;
    nrmse_bigamp = sqrt(mean((A*X - Ahat*Xhat).^2,'all')/mean((A*X).^2,'all'));
    nrmses_bigamp = [nrmses_bigamp nrmse_bigamp];    
    
    %% Run BiG-VAMP

    tstart = tic;
    [Ap, Xp]  = BiGVAMP_MC(Y, omega, nuw, N, params);
    tBiGVAMP = toc(tstart);
    ts_BiGVAMP = [ts_BiGVAMP tBiGVAMP];

    nrmse_bigvamp = sqrt(mean((A*X - Ap*Xp').^2,'all')/mean((A*X).^2,'all'));
    nrmses_bigvamp = [nrmses_bigvamp nrmse_bigvamp];
end

fprintf('=============== Results of BiG-AMP ===============\n');
fprintf(1,'nrmse BiG-AMP = %f (+/- %f)\n', mean(nrmses_bigamp), std(nrmses_bigamp));
fprintf(1,'Running time BiG-AMP = %f (+/- %f)\n', mean(ts_BiGAMP), std(ts_BiGAMP));
fprintf('=============== Results of BiG-VAMP ===============\n');
fprintf(1,'nrmse BiG-VAMP = %f (+/- %f)\n', mean(nrmses_bigvamp), std(nrmses_bigvamp));
fprintf(1,'Running time BiG-VAMP = %f (+/- %f)\n', mean(ts_BiGVAMP), std(ts_BiGVAMP));