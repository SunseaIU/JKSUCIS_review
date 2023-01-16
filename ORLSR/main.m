clear all;clc;
acc_lib=[];
lambda_lib=[];
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
    
    
    % X: d*n
    [X,X_label,Y_label,X_unlabel,Y_unlabel] = pretreat_1(X_label',Y_label,X_unlabel',Y_unlabel);
    X=[X_label,X_unlabel]; % d*n
    [d,n]=size(X);
    c=4;
    Y_L_onehot = onehot(Y_label,c);
    lambdalib = 2.^(-15:15);
    best_acc=0;
    best_lambda=-10;
    Y_predict=zeros(n_U,1);
    W_best=zeros(d,c);
    b_best=zeros(c,1);
    maxIter=50;
    for l=1:length(lambdalib)
        lambda=lambdalib(l);
        % initialize H
        H=centeringMatrix(n);
        % initialize St
        St=X*H*X';
        % initialize Y
        Y=ones(n,c)/c;
        Y(1:n_L,:)=Y_L_onehot;

        % initialize W
        W = rand(d,c); 
        % D = diag( 0.5./sqrt(sum(W.*W,2)+eps));

        for iter=1:maxIter
            % update W
            D = diag( 0.5./sqrt(sum(W.*W,2)+eps));
            A = St+lambda*D;
            B= X*H*Y;
            W=GPI(A,B);
            % update b
            b=(sum(Y,1)'-sum(W'*X,2))/n;

            % update Y
            for i=n_L+1:n
                Y(i,:)=X(:,i)'*W+b';
                Y(i,:) = EProjSimplex_new(Y(i,:));
            end

            % objective function 
            obj(iter)=F22norm(X'*W+repmat(b',[n 1])-Y)+lambda*trace(W'*D*W);

             Y_u=Y(n_L+1:n,:);
             [~,predict_label] = max(Y_u,[],2);
             acc = length(find(predict_label == Y_unlabel))./n_U;
             if acc>best_acc
                 best_acc=acc;
                 best_lambda=log2(lambda);
                 Y_predict=predict_label;
                 W_best=W;
                 b_best=b;
             end
            % fprintf('iter=%d acc=%0.4f,lambda=%d  obj=% .8f\n',iter,acc,log2(lambda),obj(iter));
        end
    end
    % fprintf('final_best_acc=%0.4f\n',best_acc);
    acc_lib=[acc_lib;best_acc];
    lambda_lib=[lambda_lib;best_lambda];
    
    fprintf('subject%d  best_acc=%0.4f ,best_lambda=%d\n',k,best_acc,best_lambda);
    path=['./results/s1_s2/sub',num2str(k),'_Y_predict.mat'];
    save(path, 'Y_predict');
    clear path;
    path=['./results/s1_s2/sub',num2str(k),'_W.mat'];
    save(path, 'W_best');
    clear path;
    path=['./results/s1_s2/sub',num2str(k),'_b.mat'];
    save(path, 'b_best');
end
path=['./results/s1_s2/','results.mat'];
save(path,'acc_lib','lambda_lib');

