%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% =======================================================================
%|********************  Class Decomposition process  ******************* |
% =======================================================================
% Step 4: create the sub-classes to construct a new dataset B using k-means
%         cluster.
% K     : the number of classes in class composition component.
%
% Input
% features_normal  ----->  noimages_norm  x 4096
% features_COVID19   -----> noimages_COVID x 4096
% features_SARS    -----> noimages_SARS x 4096
%
% Output
% Dataset B_ -------> [COVID19_1,COVID19_2, SARS_1,SARS_2,norm_1,norm_2]

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% using K-means clusering algorithm
% class decomposition component
K=2;       

idx_norm = kmeans(features_normal,K);
idx_convid19 = kmeans(features_COVID19,K);
idx_SARS = kmeans(features_SARS,K);

%%
%%%%%%%%%%%%%%%% decompostion normal classe

srcFiles_normal = dir('E:\..............\dataset A\normal\*.png');
noimages_normal=length(srcFiles_normal);

for i=1:noimages_normal
   filename_normal = strcat('E:\..............\dataset A\normal\',srcFiles_normal(i).name);
    I=imread(filename_normal); 
    
    if idx_norm(i)==1 
        imwrite(I,fullfile('E:\...............\dataset B\norm_1\',[srcFiles_normal(i).name]))
    else
        imwrite(I,fullfile('E:\...............\dataset B\norm_2\',[srcFiles_normal(i).name]))
    end
end

%%
%%%%%%%%%%%%%%%% decompostion COVID_19 classe 
srcFiles_COVID_19 = dir('E:\...............\dataset A\COVID_19\*.png'); 
noimages_COVID_19 = length(srcFiles_COVID_19);

for j=1:noimages_COVID_19
   filename_COVID_19 = strcat('E:\...............\dataset A\COVID_19\',srcFiles_COVID_19(j).name);
   I=imread(filename_COVID_19); 
    
    if idx_convid19(j)==1 
        imwrite(I,fullfile('E:\..............\dataset B\COVID_19_1\',[srcFiles_COVID_19(j).name]))
    else
        imwrite(I,fullfile('E:\..............\dataset B\COVID_19_2\',[srcFiles_COVID_19(j).name]))
    end
end


%% 
%%%%%%%%%%%%%%%% decompostion SARS classe
srcFiles_SARS   = dir('E:\...............\dataset A\SARS\*.png'); 
noimages_SARS=length(srcFiles_SARS);

for n=1:noimages_SARS
   filename_SARS = strcat('E:\...............\dataset A\SARS\',srcFiles_SARS(n).name);
    I=imread(filename_SARS); 
    
    if idx_SARS(n)==1 
        imwrite(I,fullfile('E:\...............\dataset B\SARS_1\',[srcFiles_SARS(n).name]))
    else
        imwrite(I,fullfile('E:\...............\dataset B\SARS_2\',[srcFiles_SARS(n).name]))
    end
end

