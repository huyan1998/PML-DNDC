clear;
close all;
warning('off');
rng('default');rng(0);
addpath(genpath('metrics'));
addpath(genpath('FOptMshare'));
addpath(genpath('datasets'));

load('emotions.mat');
RankingLoss = [];
HammingLoss = [];
OneError = [];
Coverage = [];
AveragePrecision = [];
for i =1:5


noisy_num = 3  ;                                                                
[pLabels,noisy_nums]=rand_noisy_num(target,noisy_num);

[RankingLoss(i,:),HammingLoss(i,:), OneError(i,:),Coverage(i,:),AveragePrecision(i,:)] = main(data, pLabels, target);

end


fprintf('RankingLoss mean=%.4f std=%.4f\n',mean(reshape(RankingLoss, [25, 1])),std(reshape(RankingLoss, [25, 1])));
fprintf('HammingLoss mean=%.4f std=%.4f\n',mean(reshape(HammingLoss, [25, 1])),std(reshape(HammingLoss, [25, 1])));
fprintf('OneError mean=%.4f std=%.4f\n',mean(reshape(OneError, [25, 1])),std(reshape(OneError, [25, 1])));
fprintf('Coverage mean=%.4f std=%.4f\n',mean(reshape(Coverage, [25, 1])),std(reshape(Coverage, [25, 1])));
fprintf('AveragePrecision mean=%.4f std=%.4f\n',mean(reshape(AveragePrecision, [25, 1])),std(reshape(AveragePrecision, [25, 1])));
