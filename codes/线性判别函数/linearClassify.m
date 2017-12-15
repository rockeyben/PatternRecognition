clear all;

% Fisher
raw_x = importdata('hw3_data.txt');
[n, d] = size(raw_x);
y = raw_x(:, d);
x = raw_x(:, 1:d-1);
x_negative = x(y ==0 ,:);
x_positive = x(y ==1 ,:);
m_negative = mean(x_negative).';
m_positive = mean(x_positive).';
S1 = cov(x_negative);
S2 = cov(x_positive);
w_fisher = inv(S1 + S2) * (m_negative - m_positive)
[n_negative, dd] = size(x_negative);
[n_positive, dd] = size(x_positive);
b_fisher = - w_fisher.' * mean(x).'

% MSE
z = [x, repmat([1], n, 1)];
a = inv(z.' * z)*z.'*y;
w_mse = a(1:3)
b_mse = a(4)

