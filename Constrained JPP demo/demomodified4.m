%This code is modified to include must-link constraints.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Copyright (c) 2014 Yahoo! Inc.
%Copyrights licensed under the MIT License. See the accompanying LICENSE file for terms.
%Author: Amin Mantrach  - amantrac at yahoo - inc dot com - http://iridia.ulb.ac.be/~amantrac/
%This is demo file on how to use the JPP decomposition,
%it will produce the final scores in terms of micro F1, macro F1, NAP and NDCG
%the data set used is the TDT2 data set publicaly available from: http://www.nist.gov/speech/tests/tdt/tdt98/index.htm, 
%We are using the matlab version available here: http://www.cad.zju.edu.cn/home/dengcai/Data/TextData.html
%The demo is configured to use 6 topics (k=6, you can change it)
%it initialize the system the first week, using NMF
%then it computes the result the remaining week from 2 to 26.
%intermediary results are displayed at each step for
%JPP, using NMF on the current timestamp (tmodel in the paper), and NMF
%on a fixed starting period timestamp (fix model).
%The demo file use lambda 10000000 this can be changed.
%In case of news, we observed that for a periof of one day, putting high
%value of lambda is the best, as we put emphasis on the past
%If you have a prior on high periodicity of the events, use value =1
%if you don't know, you can do a simple cross-val experimentation 
%a set lambda using a validation set
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% clear all
%We load the data such that we have 3 matrices
%X doc x words, T doc x time step and Y doc x label

% load TDT2.mat;
% X=fea;
% load T.mat;
% Y=[];
%load tweets1357_60.mat
X=big_data_frequency;
%load T60.mat;
dict=big_data_dict;
constwords=must_link2;
Y=[];
T=week_map;
labels=big_data_label;
k=3; %We fix the nb of top classes to track
% labels=gnd;
for v =1:3
    Y  = [Y labels==v];%concatenating result of logical test comparing labels to v
end
sumrowsY=sum(Y,2);
Ylocations=find(sumrowsY);
X = X(Ylocations,:);%finding rows of Y where sum of row of Y is nonzero, thus ignoring the rows under labels 7-10
Y = Y(Ylocations,:);
T = T(Ylocations,:);
clearvars sumrowsY;
%Ylocations now has all tweets indices which belong to labels k=1:k

%load ynewsData;
%[s i ]= sort(sum(Y),'descend');
%k=30;
%Y = Y(:,i(1:k));
%X = X(find(sum(Y,2)),:);
%Y = Y(find(sum(Y,2)),:);
%T = T(find(sum(Y,2)),:);

Wtrack=[];

 numlambda=0;

AccJPP=[];
Accbase=[];
%flag variable
JPPflag=true;
%To store NDCG scores
MR = [];
MRO = [];
MRbaseline =[];
MRfix =[];
Mmaps=[];
%To store F scores
MRF=[];
MRfixF=[];
MRbaselineF=[];
%To store MAP scores
MRMAP=[];
MRbaselineMAP=[];


regl1nmf = 0.0005;

regl1jpp = 0.05;

epsilon = 0.01;

maxiter = 10000000;
for lambda = 1E7

numlambda = numlambda+1;


%the start time period used for init of W(1) and H(1), using normal NMF
for start= 1


t = find(sum(T(:,start),2));
Xt = X(t,:);
idf = log(size(Xt,1)./(sum(Xt>0)+eps));
IDF = spdiags(idf',0,size(idf,2),size(idf,2));
Xtfidf = L2_norm_row(Xt*IDF);

%call NMF with L1 norm
[W, H] = NMF(Xtfidf, k, regl1nmf, epsilon, maxiter, false);
Hfixmodel = L2_norm_row(H);
Hbaseline2= H;
Honline=H;
HA=H;
%Wfix=L2_norm_row(W);
Wfix=W;

%number of period we consider
finT = size(T,2);


%for all the consecutive periods
for weeks = start+1:finT

fprintf('\n=========================\n');
fprintf('week number %i:\n',weeks);
fprintf('----------------\n');
%compute the grountruth as the top 10 words of the center of mass each label set    

nbtopicalwords=10;
t = find(sum(T(:,start:weeks),2));
Xt = X(t,:);
idf = log(size(Xt,1)./(sum(Xt>0)+eps));
IDF = spdiags(idf',0,size(idf,2),size(idf,2));
Xtfidf = L2_norm_row(Xt*IDF);
Yt = Y(t,:);
Htrue = Yt'*Xtfidf;
Htrue = L2_norm_row(Htrue);
[void, I]=sort(Htrue,2,'descend');
disp("initial Htrue")
H10=cell(10,3);
for c=1:3
H10(:,c)=(big_data_dict(1,I(c,1:10)))';
end
disp(H10);
for i=1:size(Htrue,1)
      Htrue(i,I(i,1:nbtopicalwords))=1;
      Htrue(i,I(i,nbtopicalwords+1:end))=0;
end
% for c=1:4
% HtrueW10=dict1234_60(1,I(c,1:10));
% disp(HtrueW10);
% end
% 


    

t = find(sum(T(:,[weeks]),2));%trying to find indices of nonzero entries in T in weeks=current week. All zero after week 13.
    Xt = X(t,:);
    idf = log(size(Xt,1)./(sum(Xt>0)+eps));
    IDF = spdiags(idf',0,size(idf,2),size(idf,2));
    Xtfidf = L2_norm_row(Xt*IDF);
    if(size(Xtfidf,1)==0)%something is happening here
        continue;
    end
     
    
    Ho=H;
    
    if(JPPflag)
      fprintf('computing JPP decomposition...');
      [W, H, M, ~] = JPPconstrained(Xtfidf, Ho, size(Ho,1), lambda, regl1jpp,  epsilon, maxiter, false,constwords,dict);
    Mmaps=[Mmaps;M];
    end
%     [void,maxHI]= max(W,[],2);
%     predictedH=[t maxHI];
    disp("H JPP")
    [void IH]=sort(L2_norm_row(H),2,'descend');
    H10=cell(10,3);
    for c=1:3
    H10(:,c)=(big_data_dict(1,IH(c,1:10)))';
    end
    disp(H10);
 
    if(numlambda==1)'
        fprintf('[ok]\ncomputing NMF decomposition...'); 
        [Wbase Hbaseline2] = NMF(Xtfidf, k,regl1nmf, epsilon, maxiter, false);
        fprintf('[ok]\n');
        Hbaseline = L2_norm_row(Hbaseline2);          
    end
   [void IH]=sort( Hbaseline,2,'descend');
   disp("Hbaseline - NMF")
    H10=cell(10,3);
    for c=1:3
    H10(:,c)=(big_data_dict(1,IH(c,1:10)))';
    end
    disp(H10);
    
    matchedH=[];
	Hev = L2_norm_row(H);
    if(JPPflag)
        Hmax = [];
        for i=1:size(Htrue,1)
         max = Htrue(i,:)*Hev(1,:)';
         maxi = 1;
         for j=2:size(Hev,1)
            val =  Htrue(i,:)*Hev(j,:)';
            if (max < val)
                max = val;
                maxi = j;
            end
         end
         matchedH=[matchedH; maxi];
         Hmax = [Hmax; Hev(maxi,:)];
        end
    end
    clearvars max;
    mappedW=W(:,matchedH);
%     [void,maxHmaxI]= max(mappedW,[],2);
%     predictedHmax=[t maxHmaxI];
    [void IHmax]=sort(Hmax,2,'descend');
    disp("Hmax - JPP")
    for c=1:3
    H10(:,c)=(big_data_dict(1,IHmax(c,1:10)))';
    end
    disp(H10);
    matchedHbase=[];
    matchedHfix=[];
    if(numlambda==1)
                Hmaxbaseline = [];
                for i=1:size(Htrue,1)
                 max = Htrue(i,:)*Hbaseline(1,:)';
                 maxi = 1;
                 for j=2:size(Hbaseline,1)
                    val =  Htrue(i,:)*Hbaseline(j,:)';
                    if (max < val)
                        max = val;
                        maxi = j;
                    end
                 end
                 matchedHbase=[matchedHbase; maxi];
                 Hmaxbaseline = [Hmaxbaseline; Hbaseline(maxi,:)];
                end
                clearvars max;
                mappedW=Wbase(:,matchedHbase);
%                 [void,maxHmaxbaseI]= max(mappedW,[],2);
%                 predictedHmaxbase=[t maxHmaxbaseI];
                [void IHmax]=sort(Hmaxbaseline,2,'descend');
                disp("Hbaseline - NMF")
                for c=1:3
                H10(:,c)=(big_data_dict(1,IHmax(c,1:10)))';
                end
                disp(H10);
                Hmaxfix = [];
                for i=1:size(Htrue,1)
                 max = Htrue(i,:)*Hfixmodel(1,:)';
                 maxi = 1;
                 for j=2:size(Hfixmodel,1)
                    val =  Htrue(i,:)*Hfixmodel(j,:)';
                    if (max < val)
                        max = val;
                        maxi = j;
                    end
                 end
                 matchedHfix=[matchedHfix; maxi];
                 Hmaxfix = [Hmaxfix; Hfixmodel(maxi,:)];
                end
                clearvars max;
                mappedW=Wfix(:,matchedHfix);
%                 [void,maxHmaxfixI]= max(mappedW,[],2);
%                 predictedHmaxfix=[t maxHmaxfixI];
                [void IHmax]=sort(Hmaxfix,2,'descend');
                disp("Hfix - originalNMF")
                for c=1:3
                H10(:,c)=(big_data_dict(1,IHmax(c,1:10)))';
                end
                disp(H10);
    end
     R=[];
    
%Wtrue is labels(Ylocations(t)), Wpred is predictedHmax(:,2)
    
%     if(JPPflag)
%          [NDCG] = performanceNDCG(Hmax,Htrue);
%         MR = [MR; NDCG];
%         fprintf('JPP  scores - NDCG: %f\n',NDCG);
%          [f_measure,accuracy]=EvaluateOrig(labels(Ylocations(t)),predictedHmax(:,2));
%          MRF=[MRF;f_measure];
%          fprintf('JPP  scores - F measure: %f\n',f_measure);
% %          APk=size(predictedHmax(:,2),1);
% %          AP=averagePrecisionAtK(labels(Ylocations(t)), predictedHmax(:,2), APk);
%          MAP=meanAveragePrecisionAtK(labels(Ylocations(t)),predictedHmax(:,2));
%          MRMAP=[MRMAP;MAP];
%          fprintf('JPP  scores - MAP measure: %f\n',MAP);
% %          fprintf('single AP score for JPP is: %f\n',AP);
%         AccJPP=[AccJPP;accuracy];
%         fprintf('JPP  scores - Accuracy: %f\n',accuracy);
%     end
%     JPPlabels=predictedH(:,2);
%     doc1=sum(JPPlabels==1);
%     doc2=sum(JPPlabels==2);
%     doc3=sum(JPPlabels==3);
%     doc4=sum(JPPlabels==4);
%     doc5=sum(JPPlabels==5);
%     Wdocs=[doc1,doc2,doc3,doc4,doc5]';
%     Wtrack=[Wtrack Wdocs];
%     
%     if (numlambda==1)
%   	  Rbaseline = [];
%   	  [NDCG] = performanceNDCG(Hmaxbaseline,Htrue);
%  	   MRbaseline = [MRbaseline;[NDCG] ];
%         fprintf('t-model  scores -  NDCG: %f\n',NDCG);
%          [f_measure,accuracy]=EvaluateOrig(labels(Ylocations(t)),predictedHmaxbase(:,2));
%          MRbaselineF=[MRbaselineF;f_measure];
%          fprintf('t-model  scores - F measure: %f\n',f_measure);
%          MAP=meanAveragePrecisionAtK(labels(Ylocations(t)),predictedHmaxbase(:,2));
%          MRbaselineMAP=[MRbaselineMAP;MAP];
%          fprintf('t-model  scores - MAP measure: %f\n',MAP);
%          Accbase=[Accbase;accuracy];
%          fprintf('JPPt-model  scores - Accuracy: %f\n',accuracy);
% 
%     
%   	  Rfix = [];
%   	  [NDCG] = performanceNDCG(Hmaxfix,Htrue);
%   	  MRfix = [MRfix;  NDCG];
%       fprintf('fix-model  scores - NDCG: %f\n',NDCG);
% %        f_measure=EvaluateOrig(labels(Ylocations(t),predictedHmaxfix));
% %        MRfixF=[MRfixF;f_measure];
% %        fprintf('fix-model  scores - F measure: %f\n',f_measure);
%       
%     end
%     fprintf('=========================\n');
%    f_measure=EvaluateOrig(labels(Ylocations(t)),predictedHmax(:,2));
end %end for weeks


end %for start 



end %for lambda
mmr = mean(MR);
fprintf('JPP Avg scores NDCG: %f\n',mmr(1));
mmr = mean(MRbaseline);
fprintf('t-model Avg NMF scores NDCG: %f\n',mmr(1));
mmr = mean(MRfix);
fprintf('fix-model Avg NMF scores NDCG: %f\n',mmr(1));

mmrF = mean(MRF);
fprintf('JPP Avg scores F: %f\n',mmrF(1));
mmrF = mean(MRbaselineF);
fprintf('t-model Avg NMF scores F: %f\n',mmrF(1));
% mmrF = mean(MRfixF);
% fprintf('fix-model Avg NMF scores F: %f\n',mmrF(1));

mmrF = mean(MRMAP);
fprintf('JPP Avg scores MAP: %f\n',mmrF(1));
mmrF = mean(MRbaselineMAP);
fprintf('t-model Avg NMF scores MAP: %f\n',mmrF(1));

mmrF = mean(AccJPP);
fprintf('JPP Avg Accuracy: %f\n',mmrF(1));
mmrF = mean(Accbase);
fprintf('t-model Avg Accuracy: %f\n',mmrF(1));


