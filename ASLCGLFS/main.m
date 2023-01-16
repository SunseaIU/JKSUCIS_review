clear all;clc;
for k=1:1
    MAX_ITER=40;
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
    
    path=['../data/subject',num2str(k),'/fea_session3_subject',num2str(k),'.mat'];
    load(path);
    [d,n_U]=size(fea);
    X_unlabel=fea;
    clear path
    path=['../data/subject',num2str(k),'/gnd_session3.mat'];
    load(path);
    Y_unlabel=gnd;
    clear fea gnd
    % X:  d*n
    [X,X_label,Y_label,X_unlabel,Y_unlabel] = pretreat_1(X_label',Y_label,X_unlabel',Y_unlabel);
    [~,Y_index]=sort([Y_label;Y_unlabel]);
    
    X=[X_label,X_unlabel]; % d*n
    [d,n]=size(X);
    c=4;
    Y_L_onehot = onehot(Y_label,c);
  
    
    D_X = getDmatrix(X');
    
    betalib = 10.^(-3:1:3);
    alphalib = 10.^(-3:1:3);
    lambdalib = 10.^(-3:1:3);
   
    
    best_acc=0;
    best_beta=0;
    best_alpha=0;
    best_lambda=0;
    
    Y_predict=zeros(n_U,c);
    
    for beta_index =1:length(betalib)
        for alpha_index=1:length(alphalib)
            for lambda_index=1:length(lambdalib)
                beta=betalib(beta_index);
                alpha=alphalib(alpha_index);
                lambda=lambdalib(lambda_index);
                
                % initialize U
                U=zeros(n);
                for i=1:n_L
                    U(i,i)=1e6;
                end
                % initialize Q and P
                Q=eye(d);
                P=eye(n);
                
                % initialize F
                F=ones(n,c)/c;
                F(1:n_L,:)=Y_L_onehot;
                Y=F;
                
                % initialize W
                W=rand(d,c);

                % initialize S
                for i = 1:n
                    S(i,:) = EProjSimplex_new((-1/(2*alpha))*D_X(i,:));
                end

                S = (S+S')/2;
                D_s = diag(sum(S)); 
                L_s = D_s - S;
                
                for iter=1:MAX_ITER
                    
                    % update Z
                    Z=(X'*W*W'*X+beta*P)\(X'*W*W'*X);
                    
                    % update P
                    for i=1:n
                        P(i,i)=0.5*norm(P(i,:),2)^(-1);
                    end
                    
                    % update F
                    F=(eye(n)+L_s+U)\(X'*W+U*onehot([Y_label;Y_unlabel],c));
                    F(1:n_L,:)=Y_L_onehot;
                    
                    % update W
                    A=X-X*Z;
                    W=(X*X'+A*A'+X*L_s*X'+lambda.*Q)\(X*F);
                    
                    % update Q
                    for i=1:d
                        Q(i,i)=0.5*norm(W(i,:),2)^(-1);
                    end

 
                    % update S
                    WX = X'*W;
                    D_WX = getDmatrix(WX);
                    D_F = getDmatrix(F);
                    T=[A;zeros(n-d,n)];
                    D_W = D_WX + D_F-2*alpha*T;
                    for i = 1:n
                        S(i,:) = EProjSimplex_new((-1/(2*alpha))*D_W(i,:));
                    end

                    % update L_s
                    S = (S+S')/2;
                    D_s = diag(sum(S)); 
                    L_s = D_s - S;


                    Y_u=F(n_L+1:n,:);
                    [~,predict_label] = max(Y_u,[],2);
                    acc = length(find(predict_label == Y_unlabel))./n_U;
                  
                    if acc>best_acc
                        best_acc=acc;
                        best_lambda=lambda;
                        best_alpha=alpha;
                        best_beta=beta;
                        Y_predict=predict_label;
                    end
                     fprintf('subject%d iter=%d acc=%0.4f \n',k,iter,acc);
                end    
                %fprintf('subject%d  acc=%0.4f alpha=%d  beta=%d lambda=%d \n',k,acc,alpha,beta,lambda);
            end   
        end
    end
    
    fprintf('subject%d  best_acc=%0.4f',k,best_acc);
    path=['./results/pre1/s1_s3/sub',num2str(k),'_Y_predict.mat'];
    save(path,'Y_predict');
    clear path;

    path=['./results/pre1/s1_s3/sub',num2str(k),'_res.mat'];
    save(path,'best_acc','best_alpha','best_beta','best_lambda');
    clear all;
end





