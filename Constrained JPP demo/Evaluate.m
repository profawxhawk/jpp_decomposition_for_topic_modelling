 function [microf_measure,Accuracy] = Evaluate(Htrue,Hmax)
% This fucntion evaluates the performance of a classification model by 
% calculating the common performance measures: Accuracy, Sensitivity, 
% Specificity, Precision, Recall, F-Measure, G-mean.
% Input: ACTUAL = Column matrix with actual class labels of the training
%                 examples
%        PREDICTED = Column matrix with predicted class labels by the
%                    classification model
% Output: EVAL = Row matrix with all the performance measures
k=size(Htrue,1);
TP=[];
TN=[];
FP=[];
FN=[];
allp=[];
alln=[];
for a=1:k
   ACTUAL=Htrue(a,:);
   PREDICTED=Hmax(a,:);
    idx = (ACTUAL()==1);

    p = length(ACTUAL(idx));
    allp=[allp;p];
    n = length(ACTUAL(~idx));
    alln=[alln;n];
    %N = p+n;

    tp = sum(ACTUAL(idx)==PREDICTED(idx));
    TP=[TP;tp];
    tn = sum(ACTUAL(~idx)==PREDICTED(~idx));
    TN=[TN;tn];
    fp = n-tn;
    FP=[FP;fp];
    fn = p-tp;
    FN=[FN;fn];

%     tp_rate = tp/p;
%     tn_rate = tn/n;

%     accuracy = (tp+tn)/N;
%     sensitivity = tp_rate;
%     specificity = tn_rate;
%     precision = tp/(tp+fp);
%     recall = sensitivity;
%     f_measure = 2*((precision*recall)/(precision + recall));
%     gmean = sqrt(tp_rate*tn_rate);

    %EVAL = [accuracy sensitivity specificity precision recall f_measure gmean];
end
sumTP=sum(TP);
sumTN=sum(TN);
sumFP=sum(FP);
sumFN=sum(FN);
sumallp=sum(allp);
sumalln=sum(alln);
precision=sumTP/(sumTP+sumFP);
recall=sumTP/(sumTP+sumFN);
microf_measure = 2*((precision*recall)/(precision + recall));
Accuracy=(sumTP+sumTN)/(sumallp+sumalln);
%EVAL=[microf_measure,Accuracy];