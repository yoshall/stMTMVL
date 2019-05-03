function mse = eval_stMTMVL_mse (Y, X, W)
%% FUNCTION eval_MTL_mse
%   computation of mean squared error given a specific model.
%   the value is the lower the better.
%
%% RELATED PAPERS, (if you use this code, please do cite our paper)
%
%   [1] Ye Liu, Yu Zheng, Yuxuan Liang, Shuming Liu, and David S. Rosenblum. 
%    "Urban Water Quality Prediction based on Multi-task Multi-view Learning.", IJCAI-16, 2016
%


task_num = length(X);
mse = 0;

total_sample = 0;
for t = 1: task_num
    y_pred = 0.5 * X{t} * W(:, t);
    mse = mse + sum((y_pred - Y{t}).^2); % length of y cancelled.
    total_sample = total_sample + length(y_pred);
end
mse = mse/total_sample;
end
