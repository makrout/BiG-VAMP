function [x1, r2,gamma_u_gauss_update] = prior_binary(r1,gamma1)
    K = size(r1,1);

    % posterior messages
    x1 = tanh(r1.*repmat(gamma1,K,1));
    alpha1 = gamma1.*mean(1-x1.^2, 1);
    
    alpha1 = max(alpha1,1e-6);
    eta1=gamma1./alpha1;
    
    % extrinsic messages
    gamma2=max(eta1-gamma1,1e-11);
    gamma_u_gauss_update=min(gamma2,1e11);
    r2=(repmat(eta1,K,1).*x1- repmat(gamma1,K,1).*r1) ./ repmat(gamma2,K,1);
    
end

