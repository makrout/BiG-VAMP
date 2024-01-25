function [mu_ext,gamma_ext,MEAN] = LMMSE_z(u, u_var, v, v_var, mu_gauss, gamma_gauss, beta)

   [m, n]= size(mu_gauss);   
   % compute the extrinsic variables
   MEAN = u * v'+trace((1/beta) * u_var * v_var')*mu_gauss* gamma_gauss;
   VAR = trace((1/beta) * u_var * v_var'*m*n + u_var * (v'*v)*m + v_var * (u'*u) * n )/m/n;
   
   % compute the extrinsic variables
   eta = (1./VAR)+gamma_gauss;
   gamma_ext = eta-gamma_gauss;
   gamma_ext=max(gamma_ext,1e-11);
   
   mu_ext = (MEAN*eta-gamma_gauss*mu_gauss)/gamma_ext;

end

   
