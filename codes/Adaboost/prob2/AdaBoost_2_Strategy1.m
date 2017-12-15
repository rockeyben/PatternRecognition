clear all;
clc;
weak_classifier_max = 4000;
% prepare data
raw_data = load('hw6_data_new.mat');
x_train_1 = raw_data.data(1:4000, 1:56);
x_train_2 = raw_data.data(5001:9000,1:56);
x_train = cat(1, x_train_1, x_train_2);
y_train_1 = ones(4000,1);
y_train_2 = ones(4000,1);
y_train_2 = -y_train_2;
y_train = cat(1, y_train_1, y_train_2);

% initialize
[n, dim] = size(x_train);
errs = []; % error of each Time step
a = []; % weight of classifiers, computed by errs
T = 0; % number of classifiers
d_list = []; % weight of samples
d = ones(n,1); % weight of samples, single step
d = d / n;
h_list = []; % weak classifiers
[xx_sort, II] = sort(x_train,'descend');

% start training
tic
while(true)
T = T + 1;

% construct simple linear classifier
row_segs = zeros(dim,1);
errs = zeros(dim,1);
for i=1:dim
    x_sort = xx_sort(:,i);
    I = II(:,i);
    y_sort = y_train(I);
    tmp_d = d(I);
    tmp_pred = ones(n,1);
    tmp_pred = -tmp_pred;
    res_0 = (tmp_pred~=y_sort);
    tmp_err = sum(res_0.*tmp_d);
    col_error = 1;
    row_idx = 1;
    for j =1:n
        tmp_pred(j)=1;
        tmp_err = tmp_err - y_sort(j)*tmp_d(j);
        if tmp_err < col_error
            col_error = tmp_err;
            row_idx=j;
        end
    end
    errs(i) = col_error;
    row_segs(i) = x_sort(row_idx);
end

[error,idx] = min(errs);
error
h_list = [h_list;row_segs(idx),idx];
a(T) = 0.5 * log((1-error)/error);
seg = row_segs(idx);
pred = sign(x_train(:,idx)-seg+0.00001);
d_list = [d_list;d];
d_new = [];
for i=1:n
    d_new(i)=d(i)*exp(-a(T)*y_train(i)*pred(i));
end
d = d_new / sum(d_new);
d = d';

if error < 0.001 || error >= 0.5 || T== weak_classifier_max
break;
end
end
toc

path1 = strcat(['.\result_prob2\hyps_',num2str(T)]);
path2 = strcat(['.\result_prob2\hyp_weights_',num2str(T)]);
path3 = strcat(['.\result_prob2\hyp_nums_',num2str(T)]);
save(path1,'h_list')
save(path2, 'a')
save(path3,'T')






