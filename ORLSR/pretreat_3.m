function [X,X_l,Y_l,X_u,Y_u] = pretreat_3(X_src,Y_src,X_tar,Y_tar)

% ÿ�����ݷ���һ��
[Y_sort,index] = sort(Y_src);
X_sort = X_src(index,:);
X_src = X_sort;
Y_src = Y_sort;
[n_l,~] = size(X_src);

[Y_sort,index] = sort(Y_tar);
X_sort = X_tar(index,:);
X_tar = X_sort;
Y_tar = Y_sort;
[n_u,~] = size(X_tar);

% ����ȥ��ֵ
X_src = X_src - repmat(mean(X_src,1),[n_l,1]);
X_tar = X_tar - repmat(mean(X_tar,1),[n_u,1]);

% ȫ������ӳ�䵽-1��1
X = [X_src;X_tar];
[X,~] = mapminmax(X',-1,1);
X = X';

X_l = X(1:n_l,:);
X_u = X(n_l+1:n_l+n_u,:);
Y_l = Y_src;
Y_u = Y_tar;

% ת������Ϊd*n
X = X';
X_l = X_l';
X_u = X_u';
