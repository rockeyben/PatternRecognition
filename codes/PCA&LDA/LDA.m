clear all;
clc;
data = importdata('swiss-data.txt');
label = data(:,1);
x = data(:, 2:4);

% LDA
x1 = x(label==1,:);
x2 = x(label==2,:);
x3 = x(label==3,:);

[n1,d] = size(x1);
[n2,d] = size(x2);
[n3,d] = size(x3);

m1 = mean(x1);
m2 = mean(x2);
m3 = mean(x3);
m = mean(x);
s_w = cov(x1)+cov(x2)+cov(x3);
s_b = n1.*(m1-m)'*(m1-m) + n2.*(m2-m)'*(m2-m) + n3.*(m3-m)'*(m3-m);
[V, D] = eig(s_b/s_w);
d = diag(D);
[sort_d,idx] = sort(d,'descend');
a1 = V(:,idx(1));
a2 = V(:,idx(2));
A = [a1, a2];

lda_res = x * A;
ld1 = lda_res(:,1);
ld2 = lda_res(:,2);
size(lda_res);
figure;
scatter(ld1,ld2)

