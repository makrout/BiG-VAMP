classdef TuneVarEstimIn < EstimIn
    % TuneVarEstimIn:  Include variance-tuning in input estimator 

    properties
        est;             % base estimator
        tuneDim = 'col'; % dimension on which the variances are to match
        nit = 4;         % number of EM iterations 
        rvarHist;        % history of rvar variances
    end
    
    methods
        % Constructor
        function obj = TuneVarEstimIn(est,varargin)
            if nargin ~= 0 % Allow nargin == 0 syntax
                obj.est = est;
                for i=1:2:length(varargin) 
                    obj.(varargin{i}) = varargin{i+1};
                end
            end
        end
        
        % Compute prior mean and variance
        function [xhat, xvar, valInit] = estimInit(obj)
            [xhat, xvar, valInit] = obj.est.estimInit;
        end
        
        % Compute posterior mean and variance from Gaussian estimate
        function [xhat, xvar] = estim(obj, rhat, rvar)

            [N,L] = size(rhat);
            NL = N*L;
            if size(rvar,1)~=N, rvar = repmat(rvar,[N 1]); end
            if size(rvar,2)~=L, rvar = repmat(rvar,[1 L]); end

            switch obj.tuneDim
              case 'joint'
                obj.rvarHist = nan(obj.nit+1,1);
                obj.rvarHist(1,:) = sum(rvar(:))/NL; 
              case 'col'
                obj.rvarHist = nan(obj.nit+1,L);
                obj.rvarHist(1,:) = (1/N)*sum(rvar,1); 
              case 'row'
                obj.rvarHist = nan(obj.nit+1,N);
                obj.rvarHist(1,:) = (1/L)*sum(rvar,2); 
              otherwise
                error('Invalid tuning dimension in TuneVarEstimIn');
            end

            rvar1 = rvar; % variance = rvar on first iteration
            for it = 1:obj.nit
                
                [xhat, xvar1] = obj.est.estim(rhat,rvar1);
                
                rvar2 = abs(xhat-rhat).^2 + xvar1;
                                
                switch obj.tuneDim
                  case 'joint'
                    rvar1 = sum(rvar2(:))/NL;
                    obj.rvarHist(it+1,1) = rvar1;
                  case 'col'
                    rvar1 = repmat(sum(rvar2)/N,[N 1]);
                    obj.rvarHist(it+1,:) = (1/N)*sum(rvar2,1);
                  case 'row'
                    rvar1 = repmat(sum(rvar2,2)/L, [1 L]);
                    obj.rvarHist(it+1,:) = (1/L)*sum(rvar2,2);
                end                
            end
            xvar = xvar1.*rvar./rvar1;
        end
        
        % Generate random samples
        function x = genRand(obj, nx)
            x = obj.est.genRand(nx);
        end
        
    end
    
end

