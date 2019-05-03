function rmse_of_method = run_stMTMV(X_train, Y_train, X_test, Y_test, paras_stMTMV, sim, ViewSegIndex_stMTMV)
%%  input
% X_train, Y_train, X_test, Y_test are the training set and testing set
% X and Y format: taskNums * {sampleNums * featureNums}
% 3 parameters in stMTMV method need to be tune, stored in paras_stMTMV
% sim is the similarity matrix which depends on the pipe attributes
% ViewSegIndex_stMTMV is the segment index between temporal view and spatial view
%% output
% rmse_of_each_method is a set of RMSE of the baseline and our algorithm
% we also create an excel file to save our result
%%
%% RELATED PAPERS, (if you use this code, please do cite our paper)
%
%   [1] Ye Liu, Yu Zheng, Yuxuan Liang, Shuming Liu, and David S. Rosenblum. 
%    "Urban Water Quality Prediction based on Multi-task Multi-view Learning.", IJCAI-16, 2016
%


rmse_of_method = [];
%% stMVMT
% three parameters mu_regMVMT, lambda_regMVMT, gamma_regMVMT need to be tune
% SimM_stMTMV is the similarity matrix
SimM_stMTMV = sim;
lambda_stMTMV = paras_stMTMV(1);
gamma_stMTMV = paras_stMTMV(2);
theta_stMTMV = paras_stMTMV(3);
STViewIndex_stMTMV = 4; % consider temporal-spatial view alignment
[W_stMTMV, fval_stMTMV] = model_stMTMVL(X_train, Y_train, SimM_stMTMV, lambda_stMTMV, gamma_stMTMV, theta_stMTMV, STViewIndex_stMTMV, ViewSegIndex_stMTMV);
MSE_stMTMV = eval_stMTMVL_mse (Y_test, X_test, W_stMTMV);
RMSE_stMTMV = eval_stMTMVL_rmse (Y_test, X_test, W_stMTMV);
fprintf('Least Squares Loss stMTMV MSE metric: %.4f, rMSE metric: %.4f \n\n', MSE_stMTMV, RMSE_stMTMV);
rmse_of_method = [rmse_of_method; RMSE_stMTMV];

