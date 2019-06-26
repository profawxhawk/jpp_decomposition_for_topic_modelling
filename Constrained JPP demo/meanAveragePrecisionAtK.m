function score = meanAveragePrecisionAtK2(actual, prediction)
%This is to measure the MAP for topic retrieval using the W matrices.

%MEANAVERAGEPRECISIONATK   Calculates the average precision at k
%   score = meanAveragePrecisionAtK(actual, prediction, k)
%
%   actual is a cell array of vectors
%   prediction is a cell array of vectors
%   k is an integer
%
%   Author: Ben Hamner (ben@benhamner.com)

%convert the parameters to cell from double
actual=num2cell(actual);
prediction=num2cell(prediction);

% if nargin<3
%     k=10;
% end
%Instead for the top k, here the entire set is used.

k=size(prediction,2);
%k=size(prediction,1);

scores = zeros(size(prediction,1),1);

for i=1:size(prediction,1)
    scores(i) = averagePrecisionAtK(actual{i}, prediction{i}, k);
end

score = mean(scores);