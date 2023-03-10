function [W]=gen_orth(X,Y,D,lam)
[~,n]=size(X);
H=eye(n)-ones(n,1)*ones(1,n)./n;
A=(X*H*X'+lam.*D)^.5;
A=max(A,A');
B=real(inv(A)'*X*H*Y);
[U,~,V]=svd(B,'econ');
Q=U*V';
W=real(inv(A)*Q);
