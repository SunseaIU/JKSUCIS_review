function [X,X_l,Y_l,X_u,Y_u] = pretreat_2(X_src,Y_src,X_tar,Y_tar)

% ÿ�����ݷ���һ��
[Y_sort,index] = sort(Y_src);
X_sort = X_src(index,:);
X_src = X_sort;
Y_src = Y_sort;


[Y_sort,index] = sort(Y_tar);
X_sort = X_tar(index,:);
X_tar = X_sort;
Y_tar = Y_sort;


% ����ȥ��ֵ ��������ӳ�䵽0��1
X_l = X_src;
[n_l,~] = size(X_l);
X_l = X_l-repmat(mean(X_l,1),[n_l,1]);
[X_l,~] = mapminmax(X_l',0,1);
X_l = X_l';
Y_l = Y_src;

X_u = X_tar;
[n_u,~] = size(X_u);
X_u = X_u-repmat(mean(X_u,1),[n_u,1]);
[X_u,~] = mapminmax(X_u',0,1);
X_u = X_u';
Y_u = Y_tar;
X = [X_l; X_u];

% ת������Ϊd*n
X = X';
X_l = X_l';
X_u = X_u';
