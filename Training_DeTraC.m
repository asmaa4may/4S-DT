%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% =======================================================================
% |         Classification of COVID-19 in chest X-ray images             |
% |                    using  DeTraC deep network                        |  
% =======================================================================
% Asmaa Abbas Hassan, Mohammed M. Abdelsamea, and Mohamed Medhat Gaber
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% This paper presents Decompose, Transfer, and Compose (DeTraC) workflow,
% for the classification of COVID-19 chest X-ray images.
% DeTraCcan deal with any irregularitiesin the image dataset by
% investigating its class boundaries using a class decomposition mechanism.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input :
%        Dataset B ----> [COVID19_1,COVID19_2, SARS_1,SARS_2,norm_1,norm_2]
% Output:
%         evaluation performance for DeTraC model
%         classifier_Accuracy      (ACC)
%         classifier_sensitivity   (SN) 
%         classifier_specifity     (SP)
%         The Area Under the Curve (AUC)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% load images
% Create an imageDataStore to read images and store images categories 
% in corresponding sub-folders.
% display the amount of samples in each class


dataset_B='E:\...............\dataset B';

dataset_B= imageDatastore(dataset_B,'IncludeSubfolders',true,...
    'FileExtensions','.png','LabelSource','foldernames',...
    'ReadFcn',@readAndPreprocessImage);

tbl = countEachLabel(dataset_B)

%% Shuffle files in ImageDatastore
dataset_B = shuffle(dataset_B);

% divide the dataset into 2 groups: 70% for trainingset and 30% for testset
[imdsTrainingSet,imdsTestSet]=splitEachLabel(dataset_B,0.7,'randomize');
numClasses = numel(categories(imdsTrainingSet.Labels));

%%
% hyper parameters for training the network
maxEpochs = 100;
miniBatchSize = 64;

opts = trainingOptions('sgdm',...
                    'Initiallearnrate',0.0001,...
                    'Minibatchsize',miniBatchSize,...   
                    'maxEpoch',maxEpochs,...            
                    'L2Regularization',0.0001,...
                    'Shuffle','every-epoch','Momentum',0.95,...
                    'Plots','training-progress','LearnRateSchedule', 'piecewise', ...    
                    'LearnRateSchedule', 'piecewise', 'LearnRateDropFactor', 0.95,'LearnRateDropPeriod',5,...
                    'CheckpointPath' ,'C:\.....................');

%% Loading an ImageNet pre-trained network
% Training the network using the training set and GPU Hardware 
% modifying the network for the new task

Img_net = input('Input the ImageNet pre-trained network: ');
if isa(Img_net,'SeriesNetwork')
    
     Layers=SeriesNet_newtask(Img_net,numClasses);
    [trainedNet,traininfo] = trainNetwork(imdsTrainingSet,Layers,opts);
   
elseif isa(Img_net,'DAGNetwork')

     lgraph=DAGNet_newtask(Img_net,numClasses);
    [trainedNet,traininfo] = trainNetwork(imdsTrainingSet,lgraph,opts);
    
end

%%
%
% Classification DeTraC model
% load mat.file from CheckpointPath to evaluate the classification performance
% Classify the testset images using the fine-tuned network,
% net:  the learned parameters

ChecKpoint_path = dir('C:\.....................\*.mat');  
num=length(ChecKpoint_path);

for i=1 : num

    
      filename = strcat('C:\.........\New folder\',srcFiles_CheckpointPath(i).name);
      load(filename);
           
      [predictedlabels,scores] = classify(net,imdsTestSet);  
      [cmat,classNames] = confusionmat(imdsTestSet.Labels, predictedlabels); 
      cm = confusionchart(cmat,classNames);
      sortClasses(cm,["Covid19_1","Covid19_2","SARS_1","SARS_2","normal_1","normal_2"])
      cmat=cm.NormalizedValues;
     
      %%  error correction equations
      % reassembling each sub-classes into the original class
      % K        : class decomposition component
      % org_classNames : clase name in dataset_A before decomposition
      % process
      
       k=2; 
       CompositionClasses= blockproc(cmat,[k k],@(block_struct) sum(block_struct.data(:)));       
       org_classNames= categorical({'Covid','SARS','normal'});
       
      [acc, sn, sp]= ConfusionMat_MultiClass (CompositionClasses,numClasses);
      
       %% creates a table from the input variables
       Evaluation_Table(i,:) = table({filename},acc,sn,sp);
        
end
                   
%% ********************* plot ROc curve *****************
  
targets=grp2idx(imdsTestSet.Labels);

[X,Y,Threshold,AUCpr] = perfcurve(targets, scores(:,1), 1, 'xCrit', 'fpr', 'yCrit', 'tpr');
plot(X,Y)
xlabel('1-specificity'); ylabel('sensitivity');
title(['ROC analysis for DeTraC (AUC: ' num2str(AUCpr) ')'])
             
    