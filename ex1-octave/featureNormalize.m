function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;
X_norm_method2 = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
%       

mu = mean(X);
sigma = std(X);
X_norm = (X - mu) ./ sigma;

% features = 1:size(X, 2);
% for i = features,
%   XminusMu  = X(:, i) - mu(i);
%   X_norm_method2(:, i) = XminusMu / sigma(i);
% end

% fprintf('X_norm method 1 = [%.0f %.0f], X_norm_method 2 [%.0f %.0f]\n', [X_norm(1:10,:) X_norm_method2(1:10,:)]');
% fprintf('Difference between methods [%.0f %.0f]\n', sum(X_norm - X_norm_method2))

% ============================================================

end
