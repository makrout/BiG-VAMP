function [mu_ext, gamma_ext] = selection(Y, S, gamma_w, mu_z_in, gamma_z_in)

    % posterior message
    mu_z_S_1 = ((gamma_w * Y + gamma_z_in * mu_z_in)/(gamma_w + gamma_z_in));
    mu_z_S_0 = mu_z_in;    
    mu_z =  S .*  mu_z_S_1 + (1-S) .* mu_z_S_0;
    
    alpha_S_1 = (gamma_z_in/(gamma_w + gamma_z_in));
    alpha_S_0 = 1;
    
    alpha = S .* alpha_S_1 + (1-S) .* alpha_S_0;
    eta = gamma_z_in/mean(alpha, 'all');   
    
    % extrinsic message
    gamma_ext = max(eta-gamma_z_in,1e-11);
    mu_ext = (eta * mu_z - gamma_z_in * mu_z_in) / gamma_ext;
    
end
