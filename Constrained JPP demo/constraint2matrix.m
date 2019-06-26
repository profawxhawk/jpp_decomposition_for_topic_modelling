function [ Q,A ] = constraint2matrix( words,dict )
%     char(words);
%     char(dict);
    delta=1;
    [nwords,ntopics]=size(words);
    ndict=size(dict,2);
    F=zeros(ndict,ntopics);
    for c=1:ntopics
        F(:,c)=ismember(dict,words(:,c));
    end
    new_size=size(F,1);
    Q=eye(new_size);
%     Q=F*F';
    A=delta*Q;
end

