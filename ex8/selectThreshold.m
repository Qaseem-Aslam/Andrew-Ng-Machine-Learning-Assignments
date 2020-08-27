function [bestEpsilon bestF1] = selectThreshold(yval, pval)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).
%

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;
m = size(pval);
actualOutliers = sum(yval);
actualNotOutliers = size(yval,1) - actualOutliers;
stepsize = (max(pval) - min(pval)) / 1000;
for epsilon = min(pval):stepsize:max(pval)
    
    % ====================== YOUR CODE HERE ======================
    % Instructions: Compute the F1 score of choosing epsilon as the
    %               threshold and place the value in F1. The code at the
    %               end of the loop will compare the F1 score for this
    %               choice of epsilon and set it to be the best epsilon if
    %               it is better than the current choice of epsilon.
    %               
    % Note: You can use predictions = (pval < epsilon) to get a binary vector 
    %       of 0's and 1's of the outlier predictions
    % 0010100000
    % 0011100000
    % 
    % 0010100000
    % 0001000000
    %
    predictions = (pval < epsilon);
    predictedOutliers = sum(predictions);
    predictedNotOutliers = size(predictions,1) - predictedOutliers;
    truePositive = sum(predictions.*yval);
    trueNegative = sum(predictions==yval) - truePositive;
    falsePositive=0;
    falseNegative=0;
    for i=1:m
        if (predictions(i) == 1) && (yval(i) == 0)
            falsePositive++;
        end
        if (predictions(i) == 0) && (yval(i) == 1)
            falseNegative++;
        end
    end

    precision = truePositive / (truePositive + falsePositive);
    recall = truePositive / (truePositive + falseNegative);

    F1 = 2 * ((precision * recall) / (precision + recall));












    % =============================================================

    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
    end
end

end
