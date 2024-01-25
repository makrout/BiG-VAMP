function [u,v] = BiVAMP(Y, var_w , r, params)
% BiVAMP algorithm for matrix factorization for Y = UV' + W
%
% Inputs
%   Y      :   observation matrix of size mxn to be factorized
%   var_w  :   noise variance
%   r      : rank of the observation matrix Y
%   params :   object containing the experiment parameters (see the Parameters object)
%
% Outputs
%   u      :   estimated U matrix
%   v      :   estimated V matrix
%

    [m,n]=size(Y);
    
    % initialize the U variable according to its prior
    switch     params.prior_u
        case    {'Gauss'}
            u = rand(m,r) ;
        case    {'Binary'}
            u = randn(m,r);
            
            mu_u_gauss_update = zeros(m,r);
            gamma_u_gauss_update = ones(1,r);
            
            mu_u_ext_old = zeros(m,r);
            gamma_u_ext_old = ones(1,r);
        case    {'Bernoulli-Gauss'}
            sparsity = params.prior_u_option.rho;
            u=randn(m,r).*(rand(m,r)<sparsity);
            
            mu_u_gauss_update = zeros(m,r);
            gamma_u_gauss_update = ones(1,r);
            
            mu_u_ext_old = zeros(m,r);
            gamma_u_ext_old = ones(1,r);
        otherwise
            error('U: unknown prior');
    end
    
    % initialize the V variable according to its prior
    switch     params.prior_v
        case    {'Gauss'}
            v=randn(n,r);
        case    {'Bernoulli-Gauss'}
            v = randn(n,r);
            mu_v_gauss_update = zeros(n,r);
            gamma_v_gauss_update = ones(1,r);
            
            mu_v_ext_old = zeros(n,r);
            gamma_v_ext_old = ones(1,r);
        otherwise
            error('V: unknown prior')
    end
    
    % initialize the estimation variables u and v
    u_old = zeros(m,r);
    v_old = zeros(n,r);
    u_var = zeros(r,r);
    v_var = zeros(r,r);
    
    % initialize the bi-LMMSE variables
    A_u = zeros(r,r);
    B_u = zeros(m,r);
    A_v = zeros(r,r);
    B_v = zeros(n,r);
    
    % iteration variables
    var_Y = sum(Y.^2, 'all')/prod(size(Y));
    damp = params.damping;
    diff = 1;
    t=0;
    
    while ((diff>params.conv_criterion))&&(t<params.nb_iter)
        % save old bi-LMMSE variables
        A_u_old = A_u;      A_v_old = A_v;
        B_u_old = B_u;      B_v_old = B_v;
        
        % update the bi-LMMSE variables
        B_u_new = (Y*v)/var_w - n*(u_old*v_var)*var_Y/var_w^2;
        A_u_new = (v'*v)/var_w + n*v_var/(params.beta*var_w) - n*v_var*var_Y/var_w^2;

        B_v_new = (Y'*u)/var_w - m*(v_old*u_var)*var_Y/var_w^2;
        A_v_new = (u'*u)/var_w + m*u_var/(params.beta*var_w) - m*u_var*var_Y/var_w^2;
        
        % save old estimates to check convergence in the end of the loop
        u_old = u;
        v_old = v;
        
        % damp the bi-LMMSE variables
        A_u = (1-damp)*A_u_old + damp*A_u_new;
        A_v = (1-damp)*A_v_old + damp*A_v_new;
        B_u = (1-damp)*B_u_old + damp*B_u_new;
        B_v = (1-damp)*B_v_old + damp*B_v_new;

        % bi-LMMSE estimate of U
        [u, u_var, mu_u_ext, gamma_u_ext] = biLMMSE_u_v(A_u,B_u, mu_u_gauss_update, gamma_u_gauss_update);
        % damp the extrinsic means/precisions from the bi-LMMSE module to the denoiser of U
        mu_u_ext = (1-damp)*mu_u_ext_old + damp*mu_u_ext;
        gamma_u_ext = (1-damp)*gamma_u_ext_old + damp*gamma_u_ext;
        mu_u_ext_old = mu_u_ext;
        gamma_u_ext_old = gamma_u_ext;
        
        % bi-LMMSE estimate of V
        [v, v_var, mu_v_ext, gamma_v_ext] = biLMMSE_u_v(A_v,B_v, mu_v_gauss_update, gamma_v_gauss_update);
        % damp the extrinsic means/precisions from the bi-LMMSE module to the denoiser of V
        mu_v_ext = (1-damp)*mu_v_ext_old + damp*mu_v_ext;
        gamma_v_ext = (1-damp)*gamma_v_ext_old + damp*gamma_v_ext;
        mu_v_ext_old = mu_v_ext;
        gamma_v_ext_old = gamma_v_ext;
        
        % the denoising module of U
        switch     params.prior_u
            case    {'Gauss'}
                % do nothing
            case    {'Binary'}
                [u_est, mu_u_gauss_update, gamma_u_gauss_update] = ...
                    prior_binary(mu_u_ext,gamma_u_ext);
            case    {'Bernoulli-Gauss'}
                [u_est, mu_u_gauss_update, gamma_u_gauss_update] = ...
                    prior_bernoulli_gauss(mu_u_ext,gamma_u_ext, 1, params.prior_u_option.rho);
        end
        
        % the denoising module of V
        switch     params.prior_v
            case    {'Gauss'}
                % do nothing
            case    {'Bernoulli-Gauss'}
                [v_est, mu_v_gauss_update, gamma_v_gauss_update] = ...
                    prior_bernoulli_gauss(mu_v_ext,gamma_v_ext, 1, params.prior_v_option.rho);
        end
        
        % Check the convergence
        diff = mean(abs(v-v_old), 'all') + mean(abs(u-u_old), 'all');
        
        t=t+1;
    end

    if strcmp(params.prior_v, 'Bernoulli-Gauss')
        v = v_est;
    end
    if strcmp(params.prior_u, 'Binary')
        u = u_est;
    end
end
