function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% see costFunctionReg in ex2
% h = sigmoid(X*theta);
h = X*theta;

% calculate cost function
thetawithout0 = [0 ; theta(2:end, :)];
penality = lambda*(thetawithout0'*thetawithout0)/(2*m);
% see costFunctionReg in ex2
% J = ((-y)'*log(h)-(1-y)'*log(1-h))/m + penality;
J = ((h - y)'*(h - y))/(2*m) + penality;

grad = (X'*(h - y) + lambda*thetawithout0)/m;

% =========================================================================

grad = grad(:);

end
