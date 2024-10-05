function [RankingLoss,HammingLoss,OneError,Coverage,AveragePrecision] = PML_test(test_data,test_target,model)

[num_test,~]=size(test_target);
[~,num_class]=size(test_target);


W = model.W;
Weights_sizepre = model.Weights_sizepre;
Bias_sizepre = model.Bias_sizepre;
% Outputs=[test_data,ones(num_test,1)]*W';

Outputs=[test_data,ones(num_test,1)]*W;
WX = test_data*W(1:end-1,:);
 Outputs1 = mapminmax(Outputs, 0, 1);
Threshold=([WX,ones(num_test,1)]*[Weights_sizepre,Bias_sizepre]')';
Pre_Labels=zeros(num_test,num_class);
for i=1:num_test
    for k=1:num_class
    %      if(Outputs(i,k)>=Threshold(1,i))
      if(Outputs1(i,k)>=0.75)
            Pre_Labels(i,k)=1;
        else
            Pre_Labels(i,k)=0;
        end
    end
end

HammingLoss=Hamming_loss(Pre_Labels',test_target');
RankingLoss=Ranking_loss(Outputs',test_target');
OneError=One_error(Outputs',test_target');
Coverage=coverage(Outputs',test_target');
AveragePrecision=Average_precision(Outputs',test_target');

end

