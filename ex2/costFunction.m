function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
sum1=0;
hypo = zeros(m,1);
for i=1:m,
	hypo(i) = sigmoid(theta' * X(i,:)');
	temp1 = y(i)*log(hypo(i));
	temp2 = (1-y(i)) * log(1-hypo(i));
	sum1 = sum1 + temp1 + temp2;
end
sum1 = -1 * sum1/m;
J=sum1;

noOfFeatures = size(X,2);
for j=1:noOfFeatures,
	sum1 = 0;
	for i=1:m,
		temp1 = (hypo(i)-y(i)) * X(i,j);
		sum1 = sum1 + temp1;
	end
	grad(j) = sum1/m;
end
% =============================================================
end
