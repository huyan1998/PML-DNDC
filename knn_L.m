function L= knn_L(train_data, k)
%KNN_L 此处显示有关此函数的摘要
%   此处显示详细说明

[num_train, ~]=size(train_data);
distance = EuDist2(train_data,train_data,1);
[near_sample , ind] = sort(distance,2);
ind = ind(:,2:k+1);
k5_distance = near_sample(:, 2:k+1);
S = zeros(num_train, num_train);
D = zeros(num_train, num_train);
L = zeros(num_train, num_train);
xita = mean(k5_distance(:,k));
K = exp(-k5_distance/(2*xita^2));
for i=1:1:num_train
%     for j=ind(i,:)
%         S()
        S(i, ind(i,:)) = K(i,:);
        
   
end

for j=1:num_train 
        D(j,j) = sum(k5_distance(j,:));
end

L = D-S ;
