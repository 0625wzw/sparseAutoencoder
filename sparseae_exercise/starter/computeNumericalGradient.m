function numgrad = computeNumericalGradient(J, theta)
% 用于计算数值梯度，即数值导数
%
% numgrad = computeNumericalGradient(J, theta)
% theta: a vector of parameters
% J: a function that outputs a real-number. Calling y = J(theta) will return the
% function value at theta.
   
% Initialize numgrad with zeros
numgrad = zeros(size(theta));
 
%% ---------- YOUR CODE HERE --------------------------------------
% Instructions:
% Implement numerical gradient checking, and return the result in numgrad. 
% (See Section 2.3 of the lecture notes.)
% You should write code so that numgrad(i) is (the numerical approximation to) the
% partial derivative of J with respect to the i-th input argument, evaluated at theta. 
% I.e., numgrad(i) should be the (approximately) the partial derivative of J with
% respect to theta(i).
%               
% Hint: You will probably want to compute the elements of numgrad one at a time.
% theta = [W1(:) ; W2(:) ; b1(:) ; b2(:)];
% W1(:) 将W1中所有数值合成一列
% A(:,1)表示矩阵A的第一列，A(1,:)表示矩阵A的第一行

EPSILON = 1e-4;
num = size(theta,1);
E = eye(num);
for i = 1:num
    delta = EPSILON * E(:,i);
    numgrad(i) = (J(theta + delta) - J(theta -delta))/2.0*EPSILON;
end
%% ---------------------------------------------------------------
end