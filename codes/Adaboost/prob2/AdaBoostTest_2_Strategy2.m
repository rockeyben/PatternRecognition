clear all;
clc;
T = 6;
weak_num = T
path1 = strcat(['.\result_prob2\hyps_',num2str(T)]);
path2 = strcat(['.\result_prob2\hyp_weights_',num2str(T)]);
path3 = strcat(['.\result_prob2\hyp_nums_',num2str(T)]);

hyp = load(path1);
h= hyp.h_list;
h_seg = h(:,1);
h_col = h(:,2);

a_list = load(path2);
a = a_list.a;

hyp_num = load(path3);
T = hyp_num.T;

raw_data = load('hw6_data_new.mat');
x_test_1 = raw_data.data(4001:5000,:);
x_test_2 = raw_data.data(9001:10000,:);
x_test = cat(1, x_test_1, x_test_2);
y_test_1 = ones(1000,1);
y_test_2 = ones(1000,1);
y_test_2 = - y_test_2;
y_test = cat(1, y_test_1, y_test_2);

pred = zeros(2000,1);
judges = zeros(T,1);
for i=1:T
    judges(i)=h_seg(i);
end

for i=1:2000
    sum = 0;
    for j=1:T 
        sum = sum + sign(x_test(i,h_col(j))-judges(j))*a(j);
    end
    pred(i) = sign(sum);
end
res = (pred~=y_test);
size(res);
sum = 0;
for i=1:2000
sum = sum + res(i);
end
accurracy = 1 - (2000-sum)/2000

