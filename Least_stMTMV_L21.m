%% FUNCTION Least_stMTMV_L21
% spatial-temporal based Multi-task Multi-view Learning model with Least Squares Loss.
%
%% OBJECTIVE
% argmin_W { sum_i^t (0.5 * norm (Y{i} - X{i}' * W(:, i))^2)
%            + pLambda * \|w^{s} * X^{s} - w^{t} * X^{t}\|_2^2 
%            + pGamma * S_{l, m} * \|w_{l} - w_{m}\|_2^2 
%            + rho1 * \|W\|_{2,1} }
%
%% INPUT
% X: {n * d} * t    : input matrix
% Y: {n * 1} * t    : output matrix
% LM                : Similarity Matrix, S_{l, m} in Eq. (3) 
% pLambda           : regularization parameter \lambda in Eq. (3)
% pGamma            : regularization parameter \Gamma in Eq. (3)
% rho_1             : L2,1-norm group Lasso parameter, regularization parameter \Theta in Eq. (3).
% SVDimInd          : index for spatial view feature vector
% optional:
%   opts.rho_L2: L2-norm parameter (default = 0).
%
%% OUTPUT
% W: model: d * t
% funcVal: function value vector.
%
%% RELATED PAPERS, (if you use this code, please do cite our paper)
%
%   [1] Ye Liu, Yu Zheng, Yuxuan Liang, Shuming Liu, and David S. Rosenblum. 
%    "Urban Water Quality Prediction based on Multi-task Multi-view Learning.", IJCAI-16, 2016
%

%% Code starts here
function [W, funcVal] = Least_stMTMV_L21(X, Y, LM, pLambda, pGamma, rho1, SVDimInd, opts)

if nargin <3
    error('\n Inputs: X, Y, rho1, should be specified!\n');
end

% initialize
task_num  = length (X);
dimension = size(X{1}, 2);


% initialize block matrix for spatial-temporal view alignment
SparseBlockMatrixP  = cell(task_num, 1); 
for k = 1:task_num;
    SparseBlockMatrixP {k} = zeros(dimension, dimension);
    featureX = X{k};
    % spatial view and temporal view
    featureX_s = featureX(:, 1:SVDimInd);
    featureX_t = featureX(:, SVDimInd+1:end);
    blockP11 = 2 * pLambda * (featureX_s)' * featureX_s;
    blockP12 = -2 * pLambda * (featureX_s)' * featureX_t;
    blockP21 = -2 * pLambda * (featureX_t)' * featureX_s;
    blockP22 = 2* pLambda * (featureX_t)'*featureX_t;
    SparseBlockMatrixP {k}(1:SVDimInd,1:SVDimInd) = blockP11;
    SparseBlockMatrixP {k}(1:SVDimInd,SVDimInd+1:end) = blockP12;
    SparseBlockMatrixP {k}(SVDimInd+1:end,1:SVDimInd) = blockP21;
    SparseBlockMatrixP {k}(SVDimInd+1:end,SVDimInd+1:end) = blockP22;    
end


X = multi_transpose(X);

if nargin <4
    opts = [];
end

% initialize options.
opts=init_opts(opts);

if isfield(opts, 'rho_L2')
    rho_L2 = opts.rho_L2;
else
    rho_L2 = 0;
end

% initialize function value
funcVal = [];


XY = cell(task_num, 1);
W0_prep = [];
for t_idx = 1: task_num
    XY{t_idx} = X{t_idx}*Y{t_idx};
    W0_prep = cat(2, W0_prep, XY{t_idx});
end

% initialize a starting point
if opts.init==2
    W0 = zeros(dimension, task_num);
elseif opts.init == 0
    W0 = W0_prep;
else
    if isfield(opts,'W0')
        W0=opts.W0;
        if (nnz(size(W0)-[dimension, task_num]))
            error('\n Check the input .W0');
        end
    else
        W0=W0_prep;
    end
end

bFlag=0; % this flag tests whether the gradient step only changes a little


Wz= W0;
Wz_old = W0;

t = 1;
t_old = 0;

iter = 0;
gamma = 1;
gamma_inc = 2;

while iter < opts.maxIter
    alpha = (t_old - 1) /t;
    
    Ws = (1 + alpha) * Wz - alpha * Wz_old;
    
    % compute function value and gradients of the search point
    gWs  = gradVal_eval(Ws);
    Fs   = funVal_eval (Ws);
    
    while true
        Wzp = FGLasso_projection(Ws - gWs/gamma, rho1 / gamma);
        Fzp = funVal_eval  (Wzp);
        
        delta_Wzp = Wzp - Ws;
        r_sum = norm(delta_Wzp, 'fro')^2;
        %         Fzp_gamma = Fs + trace(delta_Wzp' * gWs)...
        %             + gamma/2 * norm(delta_Wzp, 'fro')^2;
        Fzp_gamma = Fs + sum(sum(delta_Wzp.* gWs))...
            + gamma/2 * norm(delta_Wzp, 'fro')^2;
        
        if (r_sum <=1e-20)
            bFlag=1; % this shows that, the gradient step makes little improvement
            break;
        end
        
        if (Fzp <= Fzp_gamma)
            break;
        else
            gamma = gamma * gamma_inc;
        end
    end
    
    Wz_old = Wz;
    Wz = Wzp;
    
    funcVal = cat(1, funcVal, Fzp + nonsmooth_eval(Wz, rho1));
    
    if (bFlag)
        % fprintf('\n The program terminates as the gradient step changes the solution very small.');
        break;
    end
    
    % test stop condition.
    switch(opts.tFlag)
        case 0
            if iter>=2
                if (abs( funcVal(end) - funcVal(end-1) ) <= opts.tol)
                    break;
                end
            end
        case 1
            if iter>=2
                if (abs( funcVal(end) - funcVal(end-1) ) <=...
                        opts.tol* funcVal(end-1))
                    break;
                end
            end
        case 2
            if ( funcVal(end)<= opts.tol)
                break;
            end
        case 3
            if iter>=opts.maxIter
                break;
            end
    end
    
    iter = iter + 1;
    t_old = t;
    t = 0.5 * (1 + (1+ 4 * t^2)^0.5);
    
end

W = Wzp;

% private functions

    function [X] = FGLasso_projection (D, lambda )
    % l2.1 norm projection.
        X = repmat(max(0, 1 - lambda./sqrt(sum(D.^2,2))),1,size(D,2)).*D;
    end

% smooth part gradient.
    function [grad_W] = gradVal_eval(W)
        if opts.pFlag
            grad_W = zeros(zeros(W));
            parfor i = 1:task_num
                %grad_W (i, :) = X{i}*(X{i}' * W(:,i)-Y{i});
                grad_W (i, :) = 0.5 * X{i}*(0.5 * X{i}' * W(:,i)-Y{i}) + SparseBlockMatrixP{i}*W(:,i);
            end
        else
            grad_W = [];
            for i = 1:task_num
                %grad_W = cat(2, grad_W, X{i}*(X{i}' * W(:,i)-Y{i}) );
                temp_grad_W = 0.5* X{i}*(0.5 * X{i}' * W(:,i)-Y{i}) + SparseBlockMatrixP{i}*W(:,i);
                grad_W = cat(2, grad_W, temp_grad_W );
            end
        end
        grad_W = grad_W+ rho_L2 * 2 * W + 2 * pGamma * W * LM;
    end

% smooth part function value.
    function [funcVal] = funVal_eval (W)
        funcVal = 0;
        STView_Penalty = 0;
        if opts.pFlag
            parfor i = 1: task_num
                % spatial prediction
                spPred = X{i}(1:SVDimInd, :)'*W(1:SVDimInd, i);
                % temporal prediction
                tpPred = X{i}(SVDimInd+1:end, :)'*W(SVDimInd+1:end, i);
                % spatial-temporal prediction alignment
                STView_Penalty = STView_Penalty + norm(spPred - tpPred, 2)^2;
                
                %funcVal = funcVal + 0.5 * norm (Y{i} - X{i}' * W(:, i))^2;
                funcVal = funcVal + 0.5 * norm (Y{i} - 0.5 * X{i}' * W(:, i))^2;
            end
        else
            for i = 1: task_num
                % spatial prediction
                spPred = X{i}(1:SVDimInd, :)'*W(1:SVDimInd, i);
                % temporal prediction
                tpPred = X{i}(SVDimInd+1:end, :)'*W(SVDimInd+1:end, i);
                % spatial-temporal prediction alignment
                STView_Penalty = STView_Penalty + norm(spPred - tpPred, 2)^2;

                %funcVal = funcVal + 0.5 * norm (Y{i} - X{i}' * W(:, i))^2;
                funcVal = funcVal + 0.5 * norm (Y{i} - 0.5 * X{i}' * W(:, i))^2;
            end
        end
        funcVal = funcVal + rho_L2 * norm(W,'fro')^2 + pGamma * trace(W * LM * W') + pLambda * STView_Penalty;
    end

    function [non_smooth_value] = nonsmooth_eval(W, rho_1)
        non_smooth_value = 0;
        if opts.pFlag
            parfor i = 1 : size(W, 1)
                w = W(i, :);
                non_smooth_value = non_smooth_value ...
                    + rho_1 * norm(w, 2);
            end
        else
            for i = 1 : size(W, 1)
                w = W(i, :);
                non_smooth_value = non_smooth_value ...
                    + rho_1 * norm(w, 2);
            end
        end
    end
end