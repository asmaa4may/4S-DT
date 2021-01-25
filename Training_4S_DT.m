%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% =======================================================================
% |      4S-DT: Self Supervised Super Sample Decompositionfor            |
% |       Transfer learning in medical image classification              |  
% =======================================================================
% Asmaa Abbas Hassan, Mohammed M. Abdelsamea, and Mohamed Medhat Gaber
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% In this paper, we propose a novel deep convolutional neural network,
% we called Self Supervised Super Sample Decomposition
% for Transfer learning (4S-DT) model. 4S-DT encourages a coarse-to-fine 
% transfer learning from large-scale image recognition task to a specific 
% chest X-ray imageclassification task using a generic self-supervised sample decomposition approach
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input :
%           CXR_PretrainedNetwork.mat  : Transformation knowledge from generic CXR
%           new task from Chest x-ray dataset
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

%% load images
imdsTrainSet='E:\..................................';
imdsTrainSet= imageDatastore(imdsTrainSet,'IncludeSubfolders',true,'FileExtensions','.png','LabelSource','foldernames','ReadFcn',@readAndPreprocessImage);

imdsTestSet= 'E:\..................................';
imdsTestSet= imageDatastore(imdsTestSet,'IncludeSubfolders',true,'FileExtensions','.png','LabelSource','foldernames','ReadFcn',@readAndPreprocessImage);

numClasses = numel(categories(imdsTrainSet.Labels));

% Shuffle training and test images
imdsTrainSet = shuffle(imdsTrainSet);
imdsTestSet = shuffle(imdsTestSet);

%%
% hyper parameters for training the network
maxEpochs = 50;
miniBatchSize = 256;
numObservations = numel(imdsTrainSet.Files);
numIterationsPerEpoch = floor(numObservations / miniBatchSize);

opts = trainingOptions('sgdm',...
                    'Initiallearnrate',0.0001,...
                    'Minibatchsize',miniBatchSize,...   
                    'maxEpoch',maxEpochs,...            
                    'L2Regularization',0.001,...        
                    'Shuffle','every-epoch','Momentum',0.9,...
                    'Plots','training-progress','LearnRateSchedule', 'piecewise', ...    
                    'LearnRateSchedule', 'piecewise', 'LearnRateDropFactor', 0.9,'LearnRateDropPeriod',3,...
                    'CheckpointPath' ,'C:\.........\New folder');

%% Load the self_Supervised ImageNet for transfer learning
load('CXR_PretrainedNetwork.mat')

if isa(net,'SeriesNetwork')
    
    Layers=SeriesNet_newtask(net,numClasses);
    [trainedNet,traininfo] = trainNetwork(imdsTrainSet,Layers,opts);
      
elseif isa(net,'DAGNetwork')

    lgraph=DAGNet_newtask(net,numClasses);
    [trainedNet,traininfo] = trainNetwork(imdsTrainSet,lgraph,opts);
    
end

%% classify test images
[predictedlabels,scores] = classify(trainedNet,imdsTestSet);  
                 
%% compute the confusion matrix 
[cmat,classNames] = confusionmat(imdsTestSet.Labels, predictedlabels); 
 cm = confusionchart(cmat,classNames);
% Sort Classes
sortClasses(cm,["COVID-19","SARS","Normal"])
cmat=cm.NormalizedValues;
          
[acc, sn, sp]= ConfusionMat_MultiClass (cmat,numClasses);
classification_Accuaracy    = acc;
classification_sensitivity  = sn;
classification_specifity    = sp;

 %% ********************* plot ROc curve *****************
  
targets=grp2idx(imdsTestSet.Labels);

[X,Y,Threshold,AUCpr] = perfcurve(targets, scores(:,1), 1, 'xCrit', 'fpr', 'yCrit', 'tpr');
plot(X,Y)
xlabel('1-specificity'); ylabel('sensitivity');xlim([0 0.4]);ylim([0.5 1]);
title(['ROC analysis for 4S_DT (AUC: ' num2str(AUCpr) ')'])

