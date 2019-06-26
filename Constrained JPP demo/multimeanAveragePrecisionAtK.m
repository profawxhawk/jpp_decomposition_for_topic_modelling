function multiscore = multimeanAveragePrecisionAtK(Hactual, Hprediction)
%This is to measure the score for more than one prediction on H

%MEANAVERAGEPRECISIONATK   Calculates the average precision at k
%   score = meanAveragePrecisionAtK(actual, prediction, k)
%
%   actual is a cell array of vectors
%   prediction is a cell array of vectors
%   k is an integer
%
%   Author: Ben Hamner (ben@benhamner.com)

%convert the parameters to cell from double
numtopics=size(Hactual,1);
multiscore=[];
for a=1:numtopics
    actual=Hactual(a,:);
    prediction=Hprediction(a,:);
    actual=num2cell(actual);
    prediction=num2cell(prediction);

    % if nargin<3
    %     k=10;
    % end
    %Instead for the top k, here the entire set is used.

    k=size(prediction,1);

    scores = zeros(length(prediction),1);

    for i=1:length(prediction)
        scores(i) = averagePrecisionAtK(actual{i}, prediction{i}, k);
    end

    score = mean(scores);
    multiscore=[multiscore;score];
end
multiscore=mean(multiscore);