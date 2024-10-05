
function [RankingLoss,HammingLoss,OneError,Coverage,AveragePrecision] = main(data, pLabels, target)

target(target==-1)=0;
pLabels(pLabels==-1)=0;


% set the parameters
opt.lambda1 = 0.1;
opt.lambda2 = 0;
opt.lambda3 = 0;
opt.lambda4 = 10;


opt.max_iter = 10;
opt.k1 = 5;
opt.k2 = 2;

N = length(target);
indices = crossvalind('Kfold', 1:N ,5);  %训练集 测试集
 
for k = 1:5

test_idxs = (indices == k);
train_idxs = ~test_idxs;
        
train_data=data(train_idxs,:);
train_target=pLabels(train_idxs,:);
true_target = target(train_idxs,:);
test_data=data(test_idxs,:);test_target=target(test_idxs,:);

% pre-processing 归一化
[train_data, settings]=mapminmax(train_data');
test_data=mapminmax('apply',test_data',settings);
train_data(isnan(train_data))=0;
test_data(find(isnan(test_data)))=0;
train_data=train_data';
test_data=test_data';




model = PML_train(train_data, train_target, true_target, opt);
[RankingLoss(k),HammingLoss(k),OneError(k),Coverage(k),AveragePrecision(k)] = PML_test(test_data,test_target,model);
end

