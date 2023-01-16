clear all;clc;
k=1;

best_acc=0;
best_lambda=-10;
best_predict_label=[];

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
%  X: d*n
[X,X_label,Y_label,X_unlabel,Y_unlabel] = pretreat_1(X_label',Y_label,X_unlabel',Y_unlabel);

[d,n]=size(X);
Y=[Y_label;Y_unlabel];
c=4;
lambdalib = 2.^(-20:20);
for l=1:length(lambdalib)
    lambda=lambdalib(l);
    iters = 30;
    epsilon = 0.0001;
    [W, b] = train_mmlsr(X_label, Y_label, lambda, iters, epsilon);
    % classfication
    F_u=X_unlabel'*W+b';
    [~,predict_label] = max(F_u,[],2);
    acc = length(find(predict_label == Y_unlabel))./n_U;
    if best_acc < acc
        best_acc=acc;
        best_lambda=lambda;
        best_predict_label=predict_label;
    end
end    
path=['./results/s2_s3/sub',num2str(k),'_predict_label.mat'];
save(path,'best_predict_label');
clear path
    

 