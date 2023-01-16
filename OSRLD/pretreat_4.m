function [X,X_l,Y_l,X_u,Y_u] = pretreat_4(X_src,Y_src,X_tar,Y_tar)

% 每类数据放在一起
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

% 各自去均值
X_src = X_src - repmat(mean(X_src,1),[n_l,1]);
X_tar = X_tar - repmat(mean(X_tar,1),[n_u,1]);
X = [X_src;X_tar];

X_l = X_src;
X_u = X_tar;
Y_l = Y_src;
Y_u = Y_tar;

% 转换数据为d*n
X = X';
X_l = X_l';
X_u = X_u';
