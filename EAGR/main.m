clear all;clc;
sub=1;
path=['../data/subject',num2str(sub),'/fea_session2_subject',num2str(sub),'.mat'];
load(path);
clear path;
    
[d,n_L]=size(fea);
X_label=fea;
    
path=['../data/subject',num2str(sub),'/gnd_session2.mat'];
load(path);
Y_label=gnd;
clear fea gnd path
    
path=['../data/subject',num2str(sub),'/fea_session3_subject',num2str(sub),'.mat'];
load(path);
[d,n_U]=size(fea);
X_unlabel=fea;
clear path
path=['../data/subject',num2str(sub),'/gnd_session3.mat'];
load(path);
Y_unlabel=gnd;
clear fea gnd path

[X,X_label,Y_label,X_unlabel,Y_unlabel] = pretreat_1(X_label',Y_label,X_unlabel',Y_unlabel);
X=[X_label,X_unlabel]; % d*n
Y=[Y_label;Y_unlabel];
[d,n]=size(X);
c=4;

m=20;  % need to tune
data=X';
[IDX,anchor]=kmeans(data,m,'MaxIter',10,'emptyaction','singleton');

% Local weight estimation
[Z] = FLAE(anchor,data,3,1);

% Normalized graph Laplacian
W=Z'*Z;
Dt=diag(sum(W).^(-1/2));
S=Dt*W*Dt;
rL=eye(m,m)-S;

% Graph Regularization
label_index=ones(1,n_L);
for i=1:n_L
    label_index(1,i)=i;
end
[acc,output]= EAGReg(Z,rL, Y', label_index(1,:));


% output=output';
% predict_label=output(n_L+1:n);
% path=['./results/s2_s3/sub',num2str(sub),'_predict_label.mat'];
% save(path,'predict_label');
% clear path;
%     
% fprintf('\n The average classification accuracy of EAGR is %.2f%%.\n', mean(accuracy)*100);
