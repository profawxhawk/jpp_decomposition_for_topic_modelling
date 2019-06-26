function [DTM,dict]=countWords(tweets,stopwords)

    delimiters = {' ','$','/','.','-',':','&','*', ...          % remove those
        '+','=','[',']','?','!','(',')','{','}',',', ...
        '"','>','_','<',';','%',char(10),char(13)};
    tweets= regexprep(tweets,'RT @[^\s]*: ','');                %removes RT@ and twitter handles
    tweets = regexprep(tweets,'(http|https)[^\s]*','');         %removes urls
    tweets=lower(tweets);                                       %lowercase
    tweets = strrep(tweets, '''s', '');                         
    tweets= regexprep(tweets,'[^a-z '']',' ');                  %removes non-ascii terms
    tweets = regexprep(tweets,' +',' ');                        
%     tweets = unique(tweets);                                   % remove duplicates
    tokens = cell(tweets);                                      % cell arrray as accumulator
    numtweets=(size(tweets,1));
    for i = 1:numtweets                                         % loop over tweets
        tweet = (tweets{i});                                    % get tweet
        s = strsplit(tweet, delimiters);                        % split tweet by delimiters                                      
%        s(s == '') = [];                                       % remove empty strings
%          nums=size(s,2);
%         for z=1:nums
%             s{z}=porterStemmer(s{z});
%         end
        s(ismember(s, stopwords)) = [];                         % remove stop words
        tokens{i} = s;                                          % add to the accumulator
    end
    dict = unique([tokens{:}]);                                 % unique words
    DTM = zeros(numtweets,length(dict));                        % Doc Term Matrix
    for i = 1:numtweets                                         % loop over tokens
        [words,~,idx] = unique(tokens{i});                      % get uniqe words
        wcounts = accumarray(idx, 1);                           % get word counts
        cols = ismember(dict, words);                           % find cols for words
        DTM(i,cols) = wcounts;                                  % unpdate DTM with word counts
    end
    DTM(:,ismember(dict,{'#','@'})) = [];                       % remove # and @
    dict(ismember(dict,{'#','@'})) = [];                        % remove # and @
    numtweets
end