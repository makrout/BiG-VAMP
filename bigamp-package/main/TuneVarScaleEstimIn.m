classdef TuneVarScaleEstimIn < EstimIn
    % TuneVarScaleEstimIn:  Include variance & scale tuning in input estimator 

    properties
        est;             % base estimator
        tuneDim = 'col'; % dimension on which variances & scales are to match
        nit = 4;         % number of EM iterations 
        scaleInit = 1;   % initial scale
        keepScale = false; % overwrite scaleInit with final scale estimate?
        rvarHist;        % history of rvar variances
        scaleHist;       % history of scales 
    end
    
    methods
        % Constructor
        function obj = TuneVarScaleEstimIn(est,varargin)
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
        
        % Compute posterior mean & variance on x given rhat = beta*x + N(0,rvar)
        function [xhat, xvar] = estim(obj, rhat, rvar)

            [N,L] = size(rhat);
            NL = N*L;
            if size(rvar,1)~=N, rvar = repmat(rvar,[N 1]); end
            if size(rvar,2)~=L, rvar = repmat(rvar,[1 L]); end

            % initialize
            scale = obj.scaleInit;
            switch obj.tuneDim
              case 'joint'
                assert(length(scale)==1,'scaleInit must be a scalar')
                rhat1 = (1/scale)*rhat;
                rvar2 = sum(rvar(:))/NL;
                rvar1 = rvar2/abs(scale)^2;
                obj.scaleHist = nan(obj.nit+1,1);
                obj.scaleHist(1) = scale;
                obj.rvarHist = nan(obj.nit+1,1);
                obj.rvarHist(1,:) = sum(rvar(:))/NL; 
              case 'col'
                assert(size(scale,1)==1,'scaleInit must have 1 row')
                if size(scale,2)==1
                  scale = repmat(scale,[1 L]);
                elseif size(scale,2)~=L
                  error('scaleInit must have either 1 or L columns')
                end
                rhat1 = bsxfun(@times,1./scale,rhat);
                rvar2 = sum(rvar,1)/N;
                rvar1 = repmat(rvar2./abs(scale).^2,[N 1]);
                obj.scaleHist = nan(obj.nit+1,L);
                obj.scaleHist(1,:) = scale;
                obj.rvarHist = nan(obj.nit+1,L);
                obj.rvarHist(1,:) = (1/N)*sum(rvar,1); 
              case 'row'
                assert(size(scale,2)==1,'scaleInit must have 1 column')
                if size(scale,1)==1
                  scale = repmat(scale,[N 1]);
                elseif size(scale,1)~=N
                  error('scaleInit must have either 1 or N rows')
                end
                rhat1 = bsxfun(@times,1./scale,rhat);
                rvar2 = sum(rvar,2)/L;
                rvar1 = repmat(rvar2./abs(scale).^2,[1 L]);
                obj.scaleHist = nan(obj.nit+1,N);
                obj.scaleHist(1,:) = scale;
                obj.rvarHist = nan(obj.nit+1,N);
                obj.rvarHist(1,:) = (1/L)*sum(rvar,2); 
              otherwise
                error('Invalid tuning dimension in TuneVarScaleEstimIn');
            end                

            % run EM iterations
            for it = 1:obj.nit
                
                [xhat, xvar1] = obj.est.estim(rhat1,rvar1);
               
                switch obj.tuneDim
                  case 'joint'
                    scale = (xhat(:)'*rhat(:))...
                              /sum(abs(xhat(:)).^2 + xvar1(:)); % 1x1 
                    rvar2 = ( norm(rhat(:)-scale*xhat(:))^2 ...
                              + abs(scale)^2*sum(xvar1(:)) )/NL; % 1x1
                    obj.scaleHist(it+1) = scale;
                    obj.rvarHist(it+1) = rvar2;
                    rhat1 = (1/scale)*rhat;
                    rvar1 = rvar2/abs(scale)^2;
                  case 'col'
                    scale = sum(conj(xhat).*rhat,1)...
                             ./sum(abs(xhat).^2 + xvar1,1);%1xL 
                    rvar2 = (sum(abs(rhat-bsxfun(@times,scale,xhat)).^2,1) ...
                             + sum(xvar1,1).*abs(scale).^2 )/N;%1xL
                    obj.scaleHist(it+1,:) = scale;
                    obj.rvarHist(it+1,:) = rvar2;
                    rhat1 = bsxfun(@times,1./scale,rhat);
                    rvar1 = repmat(rvar2./abs(scale).^2,[N 1]);
                  case 'row'
                    scale = sum(conj(xhat).*rhat,2)...
                             ./sum(abs(xhat).^2 + xvar1,2);%Nx1 
                    rvar2 = (sum(abs(rhat-bsxfun(@times,scale,xhat)).^2,2) ...
                             + sum(xvar1,2).*abs(scale).^2 )/L;%Nx1
                    obj.scaleHist(it+1,:) = scale;
                    obj.rvarHist(it+1,:) = rvar2;
                    rhat1 = bsxfun(@times,1./scale,rhat);
                    rvar1 = repmat(rvar2./abs(scale).^2,[1 L]);
                end                
            end
            xvar = bsxfun(@times,xvar1.*rvar,abs(scale)./rvar2);

            % keep scale?
            if obj.keepScale
                obj.scaleInit = scale;
            end
        end
        
        % Generate random samples
        function x = genRand(obj, nx)
            x = obj.est.genRand(nx);
        end
        
    end
    
end

