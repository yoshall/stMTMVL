function [model,predictions]=MTRL(data,label,kertype,kerpar,lambda1,lambda2,testdata)
%Input:
%      data: a 1*m cell array. Each cell in 'data' contains the data matrix for one task where each row represents a data point for this task.
%      label: a 1*m cell array. Each cell in 'label' contains the labels for all data points of one task in a row vector.
%      kertype: a string. It is the kernel type with three candidaate options: 'linear', 'poly', 'rbf'.
%      kerpar: a scalar. It represents the kernel parameters with the format specified in function 'KernelFunction' in file 'CalculateKernelMatrix.m'.
%      lambda1 & lambda2: two positive scalars. They are the regularization parameters in the objective function.
%      testdata: a 1*m cell array. Each cell contains a test data matrix for one task with one row as one test data point.
%Output:
%      model: a struct variable. It contains the training model, which includes many fields, such as 
%                                'alpha', the dual variables,
%                                'b', the offset of the predictive function,
%                                'Omega', the task covariance matrix.
%      predictions: a 1*m cell array when 'testdata' is not empty. Each cell contains a row vector, which is the predictive function values for one task.
    m=length(data);%task number
    Omega=eye(m)/m;
    [data,label,task_index,ins_num]=PreprocessMTData(data,label);
    n=size(data,1);
    insIndex=cell(1,m);
    ins_indicator=zeros(m,n);
    for i=1:m
        insIndex{i}=sort(find(task_index==i));
        ins_indicator(i,insIndex{i})=1;
    end
    max_iteration=30;
    threshold=10^(-12);
    model.alpha=zeros(1,n);
    model.b=zeros(1,m);
    Km=CalculateKernelMatrix(data,kertype,kerpar);
    epsilon=10^(-8);
    m_Cor=real(Omega/(lambda1*Omega+lambda2*eye(m)));
    for t=1:max_iteration
        old_model=model;
        old_Omega=Omega;
        %Calculate alpha and b
        MTKm=Km.*m_Cor(task_index,task_index);
        model=MTRL_RR(MTKm,label,task_index,insIndex,ins_num);
        clear MTKm;
        %Calculate Omega
        temp=m_Cor(:,task_index)*diag(model.alpha);
        temp=temp*Km*temp';
        [eigVector,eigValue]=eig(temp+epsilon*eye(m));
        clear temp;
        eigValue=sqrt(abs(diag(eigValue)));
        eigValue=eigValue/sum(eigValue);
        Omega=eigVector*diag(eigValue)*eigVector';
        m_Cor=eigVector*diag(eigValue./(lambda1*eigValue+lambda2))*eigVector';
        clear eigVector eigValue;
        if norm(model.alpha-old_model.alpha,2)<=threshold*n&&norm(model.b-old_model.b,2)<=threshold*m&&norm(Omega-old_Omega,'fro')<=threshold*m*m
            clear old_model old_Omega;
            break;
        end
        clear old_model old_Omega;
    end
%     if strcmp(kertype,'linear')==1        
%         model.W=CalculateW(data,model,task_index,m,Omega,lambda1,lambda2);
%         clear W;
%     end
    model.Omega=Omega;
    model.lambda1=lambda1;
    model.lambda2=lambda2;
    clear WSquare Omega;
    if nargin==6
        predictions=[];
    else
        predictions=cell(1,m);
        for i=1:m
            if isempty(testdata{i})
                continue;
            end
            predictions{i}=Testing(testdata{i},i,data,task_index,model,kertype,kerpar,m_Cor);
        end
    end
    clear data label Omega Km task_index insIndex;
    
function prediction=Testing(testdata,task_no,data,task_index,model,kertype,kerpar,m_Cor)
    alpha=model.alpha;
    b=model.b(task_no);
    index=find(alpha~=0);
    test_Km=CalculateKernelMatrix(data(index,:),testdata,kertype,kerpar);
    prediction=(alpha(index).*m_Cor(task_no,task_index(index)))*test_Km+b;
    clear testdata task_no alpha index data task_index model kertype kerpar m_Cor;
    
function W=CalculateW(data,model,task_index,m,Omega,lambda1,lambda2)
    I=eye(m);
    W=zeros(size(data,2),m);
    for i=1:size(data,1)
        W=W+model.alpha(i)*data(i,:)'*I(task_index(i),:);
    end
    W=W*real(Omega/(lambda1*Omega+lambda2*eye(m)));
    clear data model;