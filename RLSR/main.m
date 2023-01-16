clear all;clc;

for k=1:1
    path=['../data/subject',num2str(k),'/fea_session1_subject',num2str(k),'.mat'];
    load(path);
    [d,n_L]=size(fea);
    X_label=fea;
    clear path
    path=['../data/subject',num2str(k),'/gnd_session1.mat'];
    load(path);
    Y_label=gnd;
    clear fea gnd path
    
    path=['../data/subject',num2str(k),'/fea_session2_subject',num2str(k),'.mat'];
    load(path);
    [d,n_U]=size(fea);
    X_unlabel=fea;
    clear path
     path=['../data/subject',num2str(k),'/gnd_session2.mat'];
    load(path);
    Y_unlabel=gnd;
    clear fea gnd
    %  data preprocessing d*n
    [X,X_label,Y_label,X_unlabel,Y_unlabel] = pretreat_1(X_label',Y_label,X_unlabel',Y_unlabel);
    X=[X_label,X_unlabel]; % d*n
    [d,n]=size(X);
    c=4;
    Y_L_onehot = onehot(Y_label,c);
    p=1;
    gammalib = 2.^(-10:10);
    best_acc=0;
    best_gamma=-15;
    best_W=zeros(d,c);
    for i=1:length(gammalib)
        gamma=gammalib(i);
        [ranked,theta,W,obj,Y]=RLSR(X_label,Y_L_onehot, X_unlabel,p,gamma);
         Y_u=Y(n_L+1:n,:);
        [~,predict_label] = max(Y_u,[],2);
        acc = length(find(predict_label == Y_unlabel))./n_U;
        if acc>best_acc
            best_acc=acc;
            best_gamma=gamma;
            best_W=W;
        end
        fprintf('acc=%0.4f,gamma=%d\n',acc,log2(gamma));
    end
    fprintf('subject%d  best_acc=%0.4f ,gamma=%d\n',k,best_acc,log2(best_gamma));
    clear all;
end
    


    