function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
	
	%fprintf('\n%f\t%f\tX: %f\t y: %f',theta(1,1),theta(2,1),X(1,2),y(1));
	cost1=0;
	
	for i = 1:m,
		hypothesis = (theta(1,1)+theta(2,1)*X(i,2)) - y(i);
		cost1 = cost1 + hypothesis;
	end
	cost1 = cost1/m;
	%fprintf('\nCost1: %f', cost1);
	
	
	cost2 = 0;
	for i = 1:m,
		hypothesis1 = ((theta(1,1)+theta(2,1)*X(i,2)) - y(i))*X(i,2);
		cost2 = cost2 + hypothesis1;
	end
	cost2 = cost2/m;
	%fprintf('\nCost2: %f', cost2);
	
	tempTheta1 = theta(1,1) - alpha*cost1;
	tempTheta2 = theta(2,1) - alpha*cost2;
	
	theta(1,1) = tempTheta1;
	theta(2,1) = tempTheta2;
	
	%fprintf('\n%f',theta);
	
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
	
	%fprintf('\nHistory: %f', J_history(iter));
end

end
