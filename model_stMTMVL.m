function [W, fval] = model_stMTMVL(X, Y, SimM, lambda, gamma, theta, STViewIndex, ViewSegIndex)
%% trainning wrapper for model stMTMVL (spatial-temporal multi-task multi-view learning) model
%% INPUT
%   X: {n * d} * t - input matrix
%   Y: {n * 1} * t - output matrix
%   SimM : similarity matrix for tasks (nodes)
%   lambda: spatial temporal view consistency parameter, need to be tune
%   gamma: graph Laplacian regularization parameter, need to be tune,
%   theta: L2,1-norm regularization parameter, need to be tune
%   STViewIndex: index for spatial and temporal view.
%
%   1: temporal view, 2: spatial view 3, S-T view without S-T alignment
%   4. S-T view with S-T alignment
%
%   ViewSegIndex: segment index for spatial and temporal view
%
%% OUTPUT
%   W: model: d * t
%   funcVal: function value vector.
%
%% RELATED PAPERS, (if you use this code, please do cite our paper)
%
%   [1] Ye Liu, Yu Zheng, Yuxuan Liang, Shuming Liu, and David S. Rosenblum. 
%    "Urban Water Quality Prediction based on Multi-task Multi-view Learning.", IJCAI-16, 2016
%



%% Instructions
% X, Y are 1*t cell, each cell is the trainning data and label for node (task) i.
% d is feature dimension, same for all tasks.
% lambda: spatial temporal view consistency parameter, need to be tune
% gamma: graph Laplacian regularization parameter, need to be tune,
% theta: L2,1-norm regularization parameter, need to be tune
% W is the trained model.

% number of tasks
task_num = length(X);

% graph Laplacian matrix
DM = diag(sum(SimM, 2));
LM = DM - SimM;

% generalization error parameter, rho_L2 set to be zero
% sparsityParameter
opts.rho_L2 = 0;


SVDim = ViewSegIndex;
[W, fval] = Least_stMTMV_L21(X, Y, LM, lambda, gamma, theta, SVDim, opts);

