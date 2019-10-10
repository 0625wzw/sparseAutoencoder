function [cost,grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, ...
                                             lambda, sparsityParam, beta, data)
 
% visibleSize:输入层神经单元节点数 the number of input units (probably 64)
% hiddenSize:隐藏层神经单元节点数 the number of hidden units (probably 25)
% lambda: 权重衰减系数 weight decay parameter
% sparsityParam: 稀疏性参数 The desired average activation for the hidden units (denoted in the lecture
%                           notes by the greek alphabet rho, which looks like a lower-case "p").
% beta: 稀疏惩罚项的权重 weight of sparsity penalty term
% data: 训练集 Our 64x10000 matrix containing the training data.  So, data(:,i) is the i-th training example.
% theta：参数向量，包含W1、W2、b1、b2
 
% The input theta is a vector (because minFunc expects the parameters to be a vector).
% We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this
% follows the notation convention(符号约定) of the lecture notes.
% 将长向量转换成每一层的权值矩阵和偏值向量值

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);
 
% Cost and gradient variables (your code needs to compute these values).
% Here, we initialize them to zeros.
cost = 0;
W1grad = zeros(size(W1));
W2grad = zeros(size(W2));
b1grad = zeros(size(b1));
b2grad = zeros(size(b2));
 
%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost/optimization objective J_sparse(W,b) for the Sparse Autoencoder,
%                and the corresponding gradients W1grad, W2grad, b1grad, b2grad.
%
% B = repmat (A, m,n)创建m*n的A矩阵，即B的大小为m*n个size(A)，B是m*n个A矩阵
% Jcost 直接误差
% Jweight 权值惩罚
% Jsparse 稀疏性惩罚
% [n m]= size(data);% m为样本大小，n为样本特征数

% 前向传播算法
z2 = W1*data + repmat(b1,1,m);% 将b1向量扩展至m列矩阵
a2 = sigmoid(z2);% 隐层
z3 = W2*a2 + repmat(b2,1,m);
a3 = sigmoid(z3);% 输出层，即a3=f(x)

% 计算预测产生的误差 the squared error term
Jcost = (0.5/m)*sum(sum((a3 - data).^2));
% 计算权重惩罚项  the weight decay term
Jweight = (1/2)* lambda*(sum(sum(W1.^2)) + sum(sum(W2.^2)));
% 平均活跃度 the average activation
Pj = 1/m.*sum(a2,2);% sum(a2,2) 是包含矩阵a2中每一行总和的列向量

% 计算稀疏性惩罚项 the sparsity penalty
Jsparse = beta * sum(sparsityParam.*log(sparsityParam./Pj)+...
(1-sparsityParam).*log((1-sparsityParam)./(1-Pj)));
% 损失函数cost function=Jcost+Jweight+Jsparse
cost = Jcost + Jweight + Jsparse;

% 后向传播算法
% 计算每个节点的误差值，具体参考原文提供文档公式

c3 = -(data - a3).*sigmoidGradient(z3);

% 因为加入了稀疏性规则项，所以计算偏导时需要引入该项，具体参考原文文档公式
extra_term = beta*(-sparsityParam./Pj + (1- sparsityParam)./(1 - Pj));

c2 = (W2'*c3 + repmat(extra_term,1,m)).*sigmoidGradient(z2);

W1grad = W1grad + c2*data';
W1grad = (1/m)*W1grad + lambda*W1;

W2grad = W2grad + c3*a2';
W2grad = (1/m).*W2grad + lambda*W2;

b1grad = b1grad + sum(c2,2);
b1grad = (1/m)*b1grad;

b2grad = b2grad + sum(c3,2);
b2grad = (1/m)*b2grad;

%-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).  Specifically, we will unroll
% your gradient matrices into a vector.
 
grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];
 
end
 
%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)).

% 激活函数sigmoid
function sigm = sigmoid(x)
   
    sigm = 1 ./ (1 + exp(-x));
end
% sigmoid函数逆向求导公式
function sigmoidGrad = sigmoidGradient(x)
    sigmoidGrad = sigmoid(x) .* (1 - sigmoid(x))
end