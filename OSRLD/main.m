clear all;clc;
acc_lib=[];
lambda_lib=[];

for k=1:1
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
    % X:d*n
    [X,X_label,Y_label,X_unlabel,Y_unlabel] = pretreat_1(X_label',Y_label,X_unlabel',Y_unlabel);

    [d,n]=size(X);
    c=4;
    Y_L_onehot = onehot(Y_label,c);
    lambdalib = 2.^(-20:20);
    best_acc=0;
    best_lambda=-10;
    best_Y=zeros(n,c);
    best_B=zeros(n,c);
    best_M=zeros(n,c);
    best_b=zeros(c,1);
    best_W=zeros(d,c);
    
    Y_predict=zeros(n_U,1);
    
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
        % initialize B   B=2Y-1
        B=2*Y-ones(n,c);
        for i=n_L:n
            for j=1:c
                B(i,j)=0;
            end
        end
        % initialize M
        M=zeros(n,c);
        
        % initialize W
        W = rand(d,c); 
        % D = diag( 0.5./sqrt(sum(W.*W,2)+eps));
        
        obj=[];  
        acc1=[];
        for iter=1:maxIter
            % update W
            D = diag( 0.5./sqrt(sum(W.*W,2)+eps));
            J = St+lambda*D;
            Q = X*H*(Y+B.*M);
            W=GPI(J,Q);
            % update b
            b=(sum(Y+B.*M,1)'-sum(W'*X,2))/n;

            % update Y
            temp=B.*M;
            for i=n_L+1:n
                Y(i,:)=X(:,i)'*W+b'-temp(i,:);
                Y(i,:) = EProjSimplex_new(Y(i,:));
            end
            
            % update B
            B=2*Y-ones(n,c);
            %update M
            P=X'*W+ones(n,1)*b'-Y;
            for i=1:n
                for j=1:c
                    if B(i,j)*P(i,j)>0
                        M(i,j)=P(i,j)/B(i,j);
                    else
                        M(i,j)=0;
                    end
                end
            end
            % objective function

            obj(iter)=F22norm(X'*W+repmat(b',[n 1])-Y-B.*M)+lambda*trace(W'*D*W);
          
             R=Y+B.*M;
             Y_u=R(n_L+1:n,:);
             [~,predict_label] = max(Y_u,[],2);
             acc = length(find(predict_label == Y_unlabel))./n_U;
             
             acc1(iter)=acc;
             if acc>best_acc
                 best_acc=acc;
                 best_lambda=log2(lambda);
                 Y_predict=predict_label;
                 best_Y=Y;
                 best_M=M;
                 best_B=B;
                 best_W=W;
                 best_b=b;
             end
            %fprintf('iter=%d acc=%0.4f,lambda=%d  obj=% .8f\n',iter,acc,log2(lambda),obj(iter));
        end
         path=['./results/s2_s3/obj/sub',num2str(k),'_lambda_',num2str(log2(lambda)),'_obj.mat'];
         save(path,'obj');
         clear path;
         path=['./results/s2_s3/acc/sub',num2str(k),'_lambda_',num2str(log2(lambda)),'_acc.mat'];
         save(path,'acc1');
         clear path;
    end
    acc_lib=[acc_lib;best_acc];
    lambda_lib=[lambda_lib;best_lambda];
    
    clear path;
    path=['./results/s2_s3/sub',num2str(k),'_Y_predict.mat'];
    save(path,'Y_predict');
    path=['./results/s2_s3/sub',num2str(k),'_W.mat'];
    save(path,'best_W');
    clear path;
    path=['./results/s2_s3/sub',num2str(k),'_b.mat'];
    save(path,'best_b');
    clear path;
    
    fprintf('subject%d  best_acc=%0.4f ,best_lambda=%d\n',k,best_acc,best_lambda);
    
    path=['./results/s2_s3/sub',num2str(k),'_results.mat'];
    save(path, 'best_Y', 'best_M','best_B');
    clear path;
end

path=['./results/s2_s3/','acc_res.mat'];
save(path,'acc_lib','lambda_lib');
