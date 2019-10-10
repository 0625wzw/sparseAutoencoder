function [cost,grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, ...
                                             lambda, sparsityParam, beta, data)
 
% visibleSize:������񾭵�Ԫ�ڵ��� the number of input units (probably 64)
% hiddenSize:���ز��񾭵�Ԫ�ڵ��� the number of hidden units (probably 25)
% lambda: Ȩ��˥��ϵ�� weight decay parameter
% sparsityParam: ϡ���Բ��� The desired average activation for the hidden units (denoted in the lecture
%                           notes by the greek alphabet rho, which looks like a lower-case "p").
% beta: ϡ��ͷ����Ȩ�� weight of sparsity penalty term
% data: ѵ���� Our 64x10000 matrix containing the training data.  So, data(:,i) is the i-th training example.
% theta����������������W1��W2��b1��b2
 
% The input theta is a vector (because minFunc expects the parameters to be a vector).
% We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this
% follows the notation convention(����Լ��) of the lecture notes.
% ��������ת����ÿһ���Ȩֵ�����ƫֵ����ֵ

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
% B = repmat (A, m,n)����m*n��A���󣬼�B�Ĵ�СΪm*n��size(A)��B��m*n��A����
% Jcost ֱ�����
% Jweight Ȩֵ�ͷ�
% Jsparse ϡ���Գͷ�
% [n m]= size(data);% mΪ������С��nΪ����������

% ǰ�򴫲��㷨
z2 = W1*data + repmat(b1,1,m);% ��b1������չ��m�о���
a2 = sigmoid(z2);% ����
z3 = W2*a2 + repmat(b2,1,m);
a3 = sigmoid(z3);% ����㣬��a3=f(x)

% ����Ԥ���������� the squared error term
Jcost = (0.5/m)*sum(sum((a3 - data).^2));
% ����Ȩ�سͷ���  the weight decay term
Jweight = (1/2)* lambda*(sum(sum(W1.^2)) + sum(sum(W2.^2)));
% ƽ����Ծ�� the average activation
Pj = 1/m.*sum(a2,2);% sum(a2,2) �ǰ�������a2��ÿһ���ܺ͵�������

% ����ϡ���Գͷ��� the sparsity penalty
Jsparse = beta * sum(sparsityParam.*log(sparsityParam./Pj)+...
(1-sparsityParam).*log((1-sparsityParam)./(1-Pj)));
% ��ʧ����cost function=Jcost+Jweight+Jsparse
cost = Jcost + Jweight + Jsparse;

% ���򴫲��㷨
% ����ÿ���ڵ�����ֵ������ο�ԭ���ṩ�ĵ���ʽ

c3 = -(data - a3).*sigmoidGradient(z3);

% ��Ϊ������ϡ���Թ�������Լ���ƫ��ʱ��Ҫ����������ο�ԭ���ĵ���ʽ
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

% �����sigmoid
function sigm = sigmoid(x)
   
    sigm = 1 ./ (1 + exp(-x));
end
% sigmoid���������󵼹�ʽ
function sigmoidGrad = sigmoidGradient(x)
    sigmoidGrad = sigmoid(x) .* (1 - sigmoid(x))
end