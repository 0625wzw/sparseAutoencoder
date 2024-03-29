%% CS294A/CS294W Programming Assignment Starter Code
% 稀疏自编码器程序
% Instructions
% ------------
%
% This file contains code that helps you get started on the
% programming assignment. You will need to complete the code in sampleIMAGES.m,
% sparseAutoencoderCost.m and computeNumericalGradient.m.
% For the purpose of completing the assignment, you do not need to
% change the code in this file.
%
%%======================================================================
%% STEP 0: Here we provide the relevant parameters values that will
% 第0步：提供可得到较好滤波器的相关参数值
% allow your sparse autoencoder to get good filters; you do not need to
% change the parameters below.
 
visibleSize = 8*8; % number of input units 输入层单元数
hiddenSize = 25; % number of hidden units 隐层单元数，默认为25
sparsityParam = 0.01; % desired average activation of the hidden units.稀疏值
% (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
% in the lecture notes).
lambda = 0.0001; % weight decay parameter 权重衰减系数
beta = 3; % weight of sparsity penalty term 稀疏值惩罚项权重
 
%%======================================================================
%% STEP 1: Implement sampleIMAGES 
% 第1步：实现图片采样
% After implementing sampleIMAGES, the display_network command should
% display a random sample of 200 patches from the dataset

% 图片采样程序
patches = sampleIMAGES;
% 函数display_network从训练集10000张中随机显示200张显示
display_network(patches(:,randi(size(patches,2),200,1)),8);
 
 
% Obtain random parameters theta 参数向量theta初始化
theta = initializeParameters(hiddenSize, visibleSize);
 
 
 
%%======================================================================
%% STEP 2: Implement sparseAutoencoderCost
% 第2步：实现sparseAutoencoderCost程序


% You can implement all of the components (squared error cost, weight decay term,
% sparsity penalty) in the cost function at once, but it may be easier to do
% it step-by-step and run gradient checking (see STEP 3) after each step. We
% suggest implementing the sparseAutoencoderCost function using the following steps:
%
% (1)计算cost function中均方差项、权重衰减项、惩罚项(squared error cost, weight decay term,
% sparsity penalty)
% (2)梯度检查(gradient checking)
%
% (a) Implement forward propagation in your neural network, and implement the
% squared error term of the cost function. Implement backpropagation to
% compute the derivatives. Then (using lambda=beta=0), run Gradient Checking
% to verify that the calculations corresponding to the squared error cost
% term are correct.
%
% (3)实现前向传播(forward propagation)算法和均方误差项
% (4)实现反向传播(backpropagation)算法，并计算偏导数
% (5)梯度检查(Gradient Checking)均方误差项( the squared error cost term)是否正确
%
% (b) Add in the weight decay term (in both the cost function and the derivative
% calculations), then re-run Gradient Checking to verify correctness.
%
% (6)在cost function和derivative中添加权重衰减项(weight decay)
% (7)重新运行梯度检验(Gradient Checking)来检查其正确性。
%
% (c) Add in the sparsity penalty term, then re-run Gradient Checking to
% verify correctness.
%
% （8）加入稀疏惩罚项(the sparsity penalty term)，再次运行梯度检验程序来检查其正确性。
%
% Feel free to change the training settings when debugging your
% code. (For example, reducing the training set size or
% number of hidden units may make your code run faster; and setting beta
% and/or lambda to zero may be helpful for debugging.) However, in your
% final submission of the visualized weights, please use parameters we
% gave in Step 0 above.
%
% 提升程序运行速度：首先保证核心程序均正确，然后将隐层层数减少（取决于运行环境，本人设置为2层）
% 最后，将隐层层数增加，从而获得实验需要结果。
% 另外，注意代码重用和参数重用，将重复的合并，从而达到更优效果
% 切记，一定要仔细检查，尤其梯度计算部分。

[cost, grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, lambda, ...
sparsityParam, beta, patches);
 
%%======================================================================
%% STEP 3: Gradient Checking
% 第3步：梯度检查
% Hint: If you are debugging your code, performing gradient checking on smaller models
% and smaller training sets (e.g., using only 10 training examples and 1-2 hidden
% units) may speed things up.
 
% First, lets make sure your numerical gradient computation is correct for a
% simple function. After you have implemented computeNumericalGradient.m,
% run the following:
% 梯度检查
checkNumericalGradient();
 
% Now we can use it to check your cost function and derivative calculations
% for the sparse autoencoder.
% 计算数值梯度
numgrad = computeNumericalGradient( @(x) sparseAutoencoderCost(x, visibleSize, ...
hiddenSize, lambda, ...
sparsityParam, beta, ...
patches), theta);
 
% Use this to visually compare the gradients side by side
disp([numgrad grad]);
 
% Compare numerically computed gradients with the ones obtained from backpropagation
diff = norm(numgrad-grad)/norm(numgrad+grad);
disp(diff); % Should be small. In our implementation, these values are
% usually less than 1e-9.
 
% When you got this working, Congratulations!!!
 
%%======================================================================
%% STEP 4: Train the sparse autoencoder
% 第4步：使用 minFunc (L-BFGS)训练 sparse autoencoder

% sparseAutoencoderCost is correct, You can start training your sparse
% autoencoder with minFunc (L-BFGS).
 
% Randomly initialize the parameters
theta = initializeParameters(hiddenSize, visibleSize);
 
% Use minFunc to minimize the function
addpath minFunc/
options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
% function. Generally, for minFunc to work, you
% need a function pointer with two outputs: the
% function value and the gradient. In our problem,
% sparseAutoencoderCost.m satisfies this.
options.maxIter = 400;   % Maximum number of iterations of L-BFGS to run
options.display = 'on';
 
 
[opttheta, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
visibleSize, hiddenSize, ...
lambda, sparsityParam, ...
beta, patches), ...
theta, options);
 
%%======================================================================
%% STEP 5: Visualization
% 第5步：可视化

W1 = reshape(opttheta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
display_network(W1', 12);

% 将最终结果可视化打印 weights.jpg
print -djpeg weights.jpg % save the visualization to a file