function model = PML_train(train_data, train_target, true_target, opt)

lambda1 = opt.lambda1;
lambda2 = opt.lambda2;
lambda3 = opt.lambda3;
lambda4 = opt.lambda4;
max_iter = opt.max_iter;
k1 = opt.k1;
k2 = opt.k2;

model = [];
[num_train,dim]=size(train_data);
[~,num_label]=size(train_target);

K = num_label;
%% Training

X = [train_data, ones(num_train,1)];
% X = train_data;
Y =  train_target;
U =abs( randn(num_label,K));
V =abs( randn(num_train,K));
W =randn(dim+1,num_label);


L=knn_L(X, k1);


for t = 1:max_iter
    
    
    
    
    % Update V

     V = lyap(lambda2*L,(1 + lambda1)*(U)'*U , -(Y*U + lambda1*X*W*U));
      
      %V = (Y*U + lambda1*X*W*U)/((1 + lambda1)*(U)'*U);
      
      

    %update U

    U = (Y'*V + lambda1*W'*(X)'*V)/((1 + lambda1)*(V)'*V);
    
   
    
    
    LU=knn_L(U, k2);

   
    %update W
      
    W = lyap(lambda1*(X)'*X , lambda3*LU+ lambda4*eye(num_label), -lambda1*(X)'*V*(U)');
    
    
    
end

%% Computing the size predictor using linear least squares modelO
Outputs = X*W;

Left=Outputs;

Right=zeros(num_train,1);
for i=1:num_train
    temp=Left(i,:);
    [temp,index]=sort(temp);
   
    candidate=zeros(1,num_label+1);
 
    candidate(1,1)=temp(1)-0.1;
    for j=1:num_label-1
        candidate(1,j+1)=(temp(j)+temp(j+1))/2;
    end
    candidate(1,num_label+1)=temp(num_label)+0.1;
    miss_class=zeros(1,num_label+1);
    for j=1:num_label+1
        temp_notlabels=index(1:j-1);
        
        temp_labels=index(j:num_label);
        [~,false_neg]=size(setdiff(temp_notlabels,find(true_target(i,:)==0)));
        [~,false_pos]=size(setdiff(temp_labels,find(true_target(i,:)==1)));
        miss_class(1,j)=false_neg+false_pos;
    end
    [~,temp_index]=min(miss_class);
    Right(i,1)=candidate(1,temp_index);
end
 
Left=[Left,ones(num_train,1)];

tempvalue=(Left\Right)';
Weights_sizepre=tempvalue(1:num_label);
Bias_sizepre=tempvalue(num_label+1);

model.W = W;
model.Weights_sizepre=Weights_sizepre;
model.Bias_sizepre=Bias_sizepre;  


