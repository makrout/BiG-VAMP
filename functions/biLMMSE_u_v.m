function [MEAN, VAR, mu_ext,gamma_ext] = biLMMSE_u_v(A,B,mu_gauss, gamma_gauss)

   K = size(mu_gauss,1);
   % posterior message
   VAR=inv(diag(gamma_gauss).*eye(size(A))+A);
   MEAN=(B+mu_gauss.*repmat(gamma_gauss, K, 1))*VAR;
   
   % extrinsic message
   eta = (1./diag(VAR))';
   gamma_ext = (eta-gamma_gauss);
   gamma_ext =max(gamma_ext,1e-5);
   mu_ext = (repmat(eta,K,1) .* MEAN - repmat(gamma_gauss,K,1) .* mu_gauss)./ repmat(gamma_ext,K,1);
end

