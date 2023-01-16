clear all;clc;
for k=1:1
    MAX_ITER=50;
    epsilon=1e-5;
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
    % X: d*n
    [X,X_label,Y_label,X_unlabel,Y_unlabel] = pretreat_1(X_label',Y_label,X_unlabel',Y_unlabel);
    X=[X_label,X_unlabel]; % d*n
    [d,n]=size(X);
    c=4;
    Y_L_onehot = onehot(Y_label,c);
    
    H = eye(n) - ones(n)/n;
    XHX=X*H*X';
    
    gammalib = 2.^(-20:20);
    best_acc=0;
    best_gamma=-15;
    best_W=zeros(d,c);
    Y_predict=zeros(n,c);
    
    for i=1:length(gammalib)
        gamma=gammalib(i);
        % initialize Y
        Y=ones(n,c)/c;
        Y(1:n_L,:)=Y_L_onehot;
        % initialize M
        M=zeros(n,c);
        % initialize bigThea
        bigTheta=eye(d)/d;
        % Update process
         for iter=1:MAX_ITER
            Y_temp=(2*Y-ones(n,c)).*M;
            XHY=X*H*(Y+Y_temp);
            % update W
            W=(XHX+gamma*(bigTheta^-2))\XHY;
            q=1;
            temp=sum(W.*W,2).^(1/(q+1))+epsilon;
            % update theta
            bigTheta=diag( temp/ sum(temp) ).^(q/2);
            % update b
            Y_temp=Y+(2*Y-ones(n,c)).*M;
            b=(sum(Y_temp,1)'-sum(W'*X,2))/n;
            % update M
            temp=(2*Y-ones(n,c)).*(X'*W+ones(n,1)*b'-Y);
            M=max(temp,zeros(n,c));

            % update Y
            for j=n_L+1:n
               e=W'*X(:,j)+b+M(j,:)';
               a=2*M(j,:)'+ones(c,1);
               [y,~]=fun2(e,a);
               Y(j,:)=y';
            end
            % prediction accuracy
             R=Y+(2*Y-ones(n,c)).*M;
             Y_u=R(n_L+1:n,:);
             [~,predict_label] = max(Y_u,[],2);
             acc = length(find(predict_label == Y_unlabel))./n_U;
             % fprintf('iter=%d,acc=%0.4f,gamma=%d\n',iter,acc,log2(gamma));
             if acc>best_acc
                 best_acc=acc;
                 best_gamma=log2(gamma);
                 Y_predict=predict_label;
             end
         end
    end
    fprintf('subject%d  best_acc=%0.4f ,gamma=%d\n',k,best_acc,log2(best_gamma));
    path=['./results/pre1/s1_s2/sub',num2str(k),'_Y_predict.mat'];
    save(path,'Y_predict');
    clear path;
    
    path=['./results/pre1/s1_s2/sub',num2str(k),'_acc.mat'];
    save(path,'best_acc');
    clear all;
end