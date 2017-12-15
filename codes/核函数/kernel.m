clear all;

% process data
raw_data = importdata('hw5_data.txt');

x = raw_data(:, 1:3);
y = raw_data(:, 4);

x_1 = x(y==0,:);
x_2 = x(y==1,:);
y_1 = y(y==0,:);
y_2 = y(y==1,:);

x_1_train = x_1(1:40,:);
x_2_train = x_2(1:40,:);
y_1_train = y_1(1:40,:);
y_2_train = y_2(1:40,:);
x_1_test = x_1(41:50, :);
x_2_test = x_2(41:50, :);
y_1_test = y_1(41:50,:);
y_2_test = y_2(41:50,:);
x_train = cat(1, x_1_train, x_2_train);
y_train = cat(1, y_1_train, y_2_train);
x_test = cat(1, x_1_test, x_2_test);
y_test = cat(1, y_1_test, y_2_test);

[n, d] = size(x_train);
[n1, d] = size(x_1_train);
[n2, d] = size(x_2_train);
[np, d] = size(x_test);

% some important variables
N1 = zeros(n,n);
N2 = zeros(n,n);
K = zeros(n,n);
N = zeros(n,n);
Gamma = zeros(n,1);
a = zeros(n,1);
b = 0;

% some hyper params
t = 1;

% start to train
for i=1:n
for j=1:n
K(i,j) = rbf(x_train(i,:),x_train(j,:));
end
end


for i=1:n
for j=1:n
tmp_i = K(i,1:n1);
tmp_j = K(j,1:n1);
tmp_i = tmp_i - mean(tmp_i);
tmp_j = tmp_j - mean(tmp_j);
N1(i,j)=sum(tmp_i.*tmp_j);

tmp_i = K(i,n1+1:n);
tmp_j = K(j,n1+1:n);
tmp_i = tmp_i - mean(tmp_i);
tmp_j = tmp_j - mean(tmp_j);
N2(i,j)=sum(tmp_i.*tmp_j);
end
end

N = N1 + N2;

for i=1:n
tmp_1 = K(i,1:n1);
tmp_2 = K(i,n1+1:n);
Gamma(i) = mean(tmp_1)-mean(tmp_2);
end

a = inv(N+t*K)*Gamma;

for i=1:n
tmp_1 = K(i,1:n1);
tmp_2 = K(i,n1+1:n);
b = b + a(i)*(mean(tmp_1)+mean(tmp_2));
end

b = b * (-0.5);

% Testing
pred = zeros(np,1);

for i=1:np
for j=1:n
pred(i) = pred(i) + a(j)*rbf(x_train(j,:),x_test(i,:));
end
end

pred = pred + b

pred(pred > 0) = 0; % belongs to class 1
pred(pred < 0) = 1; % belongs to class 2

res = zeros(np,1);
res(pred==y_test) = 1;

% final accuracy
accuracy = sum(res)/np


function y = rbf(x1, x2)
    sigma = 2;
    gamma = 1 / (2 * sigma^2);
    y = exp(-gamma * sum((x1-x2).*(x1-x2)));
end