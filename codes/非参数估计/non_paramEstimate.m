x = importdata('hw2_data.txt') ;
[n, m] = size(x);

% 1. parzen window method
h = [100, 1000, 10000];
colors=['r-', 'g-', 'm-'];
h_len = size(h, 2);
figure;
hold on;
x_0 = linspace(min(x), max(x));
x_1 = repmat(x_0, n, 1);
for i=1 : h_len
    y_0 = sum(K((x_1 - x)/h(i)))/(n*h(i));
    fig = plot(x_0, y_0, char(colors(i)));
    set(fig,'linewidth', 2);
end
fig = legend('h_n = 100', 'h_n = 1000', 'h_n = 10000');
set(fig, 'fontsize', 10);
% print('hw2_parzen_res', '-dpng');

% 2. k_n neighbour method
k = [3, 5, 10];
k_len = size(k, 2);
figure;
hold on;
for i = 1 : k_len
    y_1 = zeros(size(x_0));
    for j = 1: size(x_0, 2)
        dist = abs(x_0(j)-x);
        [sort_dist, index] = sort(dist);
        k_n_neighbour = x(index(1:k(i)));
        v = max(k_n_neighbour) - min(k_n_neighbour);
        y_1(j) = k(i)/(n*v);
    end
    fig2 = plot(x_0, y_1, char(colors(i)));
    set(fig2, 'linewidth', 2);
end
fig2 = legend('k_n = 3', 'k_n = 5', 'k_n = 10');
set(fig2, 'fontsize', 10);
% print('hw2_kn_res', '-dpng');

function y = K(xi)
    y = exp(-0.5 * xi.^2) / (2*pi)^0.5; 
end



