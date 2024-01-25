function [x_hat, r_hat_ext, gamma_ext] = prior_bernoulli_gauss(r_hat,gamma, signal_var,rho)

    r_hat = r_hat';
    gamma = gamma';
    
    [batch_size, Nu] = size(r_hat);
    
    mu_1_r = r_hat;
    nu_1_r = 2./gamma;
    
    mu_2_x = zeros(batch_size,Nu);
    nu_2_x= 2*signal_var;
    
    mu_r_x = (nu_2_x*mu_1_r + repmat(nu_1_r, 1, Nu).*mu_2_x)./(repmat(nu_1_r, 1, Nu) + nu_2_x);
    nu_r_x = (nu_1_r*nu_2_x)./(nu_1_r + nu_2_x);
    
    
    Ax_1 = (rho*mu_r_x./(sqrt(repmat(nu_1_r, 1, Nu) + nu_2_x))).*exp(-(abs(mu_1_r - mu_2_x).^2)./(repmat(nu_1_r, 1, Nu) + nu_2_x));
      
    
    Ax_2 = (((repmat(nu_r_x, 1, Nu)./2 + mu_r_x.^2)*rho)./sqrt(repmat(nu_1_r, 1, Nu) + nu_2_x)).*...
                                                   exp(-(abs(mu_1_r - mu_2_x).^2)./(repmat(nu_1_r, 1, Nu) + nu_2_x));                                           
    
    Bx_1 = ((1 - rho)./sqrt(repmat(nu_1_r, 1, Nu))).*exp(-(abs(mu_1_r).^2)./(repmat(nu_1_r, 1, Nu))) + ...
                                (rho./sqrt(repmat(nu_1_r, 1, Nu) + nu_2_x)).*exp(-(abs(mu_1_r - mu_2_x).^2)./(repmat(nu_1_r, 1, Nu) + nu_2_x));
    
    Bx_1 = max(Bx_1, 10^(-11));

    x_hat = Ax_1./Bx_1;
    
    Q_x = Ax_2./Bx_1 - x_hat.^2;
    
    alpha = gamma.*mean(Q_x, 2);
    
    % Add the extrinsic calculation
    eta = gamma./alpha;
    gamma_ext = eta - gamma;
    gamma_ext = min(max(gamma_ext, 10^(-11)),10^(11));
    
    r_hat_ext = (repmat(eta, 1, Nu).*x_hat - repmat(gamma, 1, Nu).*r_hat)./repmat(gamma_ext, 1, Nu);
    
    % Transpose for dimension compatibility
    x_hat = x_hat';
    r_hat_ext = r_hat_ext';
    gamma_ext = gamma_ext';
        
end