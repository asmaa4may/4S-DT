%%%%%%%%%%%%%%%% Extract the learned Features from each image Using CNN 

% load the checkpointtrained network that could be descriminated between the three
% classes. 
% Determine the layer number and the number of features. 
%In our work laye= AlexNet and FeaturesNum=4096 .
% load the images within each class and store each one as imageDatastore 
% specify two separeted variables for each class to contain the activations
% features from the ith image.

% Input: 
% 'E:\..................\convnet_checkpoint__   .mat')

% Output :
% features_normal  ----->  noimages_norm  x 4096
% features_COVID19   -----> noimages_COVID x 4096
% features_SARS    -----> noimages_SARS x 4096

%% load previous pre-training work
load('net_checkpoint__...........................') 
layer=net.Layers(end-2);            
FeaturesNum=(layer.InputSize);

%% %%%%%%%%%%%%%%%% load covide_19 images%%%%%%%%%%%%%%%%
covide19_images=dir('E:\.................\dataset_A\Covid_19\*.png');
noimages_covide19=length(covide19_images);

features_covide19=zeros(noimages_covide19,FeaturesNum);   
  
for i=1 : noimages_covide19
    
    filename_covide19 = strcat('E:\.................\dataset_A\Covid_19\',covide19_images(i).name);
    covide_image=imageDatastore(filename_covide19);
    covide_image.Labels='Covid_19';
    covide_image.Labels=categorical(covide_image.Labels);
    covide_image.ReadFcn= @(filename)readAndPreprocessImage(filename);
         
    features_covide19(i,:)=activations(net,covide_image,layer,'OutputAs','rows');
    
end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%% load normal images%%%%%%%%%%%%%%%%
normal_images=dir('E:\.................\dataset_A\normal\*.png');
noimages_norm=length(normal_images);

features_normal=zeros(noimages_norm,FeaturesNum);  

 for j=1 : noimages_norm
    
    filename_normal = strcat('E:\.................\dataset_A\normal\',normal_images(j).name);
    norm_image=imageDatastore(filename_normal);
    norm_image.Labels='normal';
    norm_image.Labels=categorical(norm_image.Labels);
    norm_image.ReadFcn= @(filename)readAndPreprocessImage(filename);
   
    features_normal(j,:) = activations(net,norm_image,layer,'OutputAs','rows');
       
 end
 
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%% load SARS images%%%%%%%%%%%%%%%%
SARS_images=dir('E:\.................\dataset_A\SARS\*.png');
noimages_SARS=length(SARS_images);

features_SARS=zeros(noimages_SARS,FeaturesNum);  

 for j=1 : noimages_SARS
    
    filename_SARS = strcat('E:\.................\dataset_A\SARS\',SARS_images(j).name);
    SARS_image=imageDatastore(filename_SARS);
    SARS_image.Labels='SARS';
    SARS_image.Labels=categorical(SARS_image.Labels);
    SARS_image.ReadFcn= @(filename)readAndPreprocessImage(filename);
   
    features_SARS(j,:) = activations(net,SARS_image,layer,'OutputAs','rows');
       
 end 
