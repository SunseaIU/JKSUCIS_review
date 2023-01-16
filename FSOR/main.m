clear all;clc;
k=1;
path=['../data/subject',num2str(k),'/fea_session2_subject',num2str(k),'.mat'];
load(path);
[d,n_L]=size(fea);
 X_label=fea;
 clear path
 path=['../data/subject',num2str(k),'/gnd_session2.mat'];
 load(path);
 Y_label=gnd;
 clear fea gnd path
    
 path=['../data/subject',num2str(k),'/fea_session3_subject',num2str(k),'.mat'];
 load(path);
 [d,n_U]=size(fea);
 X_unlabel=fea;
 clear path
 path=['../data/subject',num2str(k),'/gnd_session3.mat'];
 load(path);
 Y_unlabel=gnd;
 clear fea gnd path
  % X: d*n
 [X,X_label,Y_label,X_unlabel,Y_unlabel] = pretreat_4(X_label',Y_label,X_unlabel',Y_unlabel);
  
 [d,n]=size(X);
 c=4;
 Y_label_onehot=onehot(Y_label,c);
 Y_label_onehot=Y_label_onehot';
 H=centeringMatrix(n_L);
    
% initialize theta
maxIter=50;
[W,b]=least_squares_regression(X_label,Y_label_onehot,10);
best_acc=0;
best_predict_label=zeros(n_U,c);
best_W=zeros(d,c);

for iter=1:maxIter
    % update W
    A=X_label*H*X_label';
    B=X_label*H*Y_label_onehot';
    W=GPI(A,B);
    b=1/n*(Y_label_onehot*ones(n_L,1)-W'*X_label*ones(n_L,1));
    F_u=X_unlabel'*W+b';
    [~,predict_label] = max(F_u,[],2);
    acc = length(find(predict_label == Y_unlabel))./n_U;
    if best_acc<acc
        best_acc=acc;
        best_predict_label=predict_label;
        best_W=W;
    end
end





