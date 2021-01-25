function [all_ACC,all_sn, all_sp] = ConfusionMat_MultiClass(cmat,numClasses)

ACC_Class=zeros(1,numClasses);       % Accuracy
SN_Class=zeros(1,numClasses);        % Sensitivity
SP_Class=zeros(1,numClasses);        % Specificity


for C=1:numClasses

    TP=0;   TN=0;   FP=0;  FN=0;
    
 %%%%%%%%%%%%%%%%% compute TP

            TP =TP +cmat(C,C);
                

%%%%%%%%%%%%%%%%% compute FN
             i=C;
                 for j=1:numClasses
                     if j ~= i 
                      FN =FN +cmat(i,j);
                     end
                 end
            

%%%%%%%%%%%%%%%%% compute FP
             i=C;
                 for j=1:numClasses
                     if j ~= i
                       FP =FP +cmat(j,i);
                     end
                 end
           
%%%%%%%%%%%%%%%%% compute TN
            for i=1:numClasses
                if i ~= C
                   for j=1:numClasses
                        if j ~= C
                             TN= TN +cmat(i,j);
                        end
                   end
                end
            end
 

ACC_Class(1,C)=(TP+TN)/(TP+TN+FP+FN);
SN_Class(1,C) = TP / (TP + FN);        
SP_Class(1,C)= TN /(TN + FP); 
            
end
    

all_ACC= mean(ACC_Class) ;
all_sn= mean(SN_Class) ;
all_sp= mean(SP_Class) ;

end
