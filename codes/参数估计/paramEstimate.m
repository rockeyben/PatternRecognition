% (1) 
x = importdata('A.txt');
[n, m] = size(x);

theta = sum(log(x)) / n
sigma = sqrt(sum((log(x)-theta).^2)/n)

figure;
hold on;
title('given distribution');
y = zeros(n, 1);
plot(x, y, 'g.');
x_0 = linspace(min(x), max(x));
y_0 = p(x_0, theta, sigma);
axis([min(x) max(x) 0 0.00005])
plot(x_0, y_0, 'r');
print('res_1','-dpng');
% (2)
figure;
hold on;
title('gaussian distribution');
miu = mean(x);
sigma2 = sqrt(sum((x-miu).^2)/n);
plot(x, y, 'g.')
x_1 = linspace(min(x), max(x));
y_1 = p_gaussian(x_1, miu, sigma2);
plot(x_1, y_1, 'r')
print('res_2','-dpng')
function y = p(x, theta, sigma)
    y = exp(-(log(x)-theta).^2/(2*sigma^2)) ./ (sigma*x*(2*pi)^0.5);
end

function y = p_gaussian(x, miu, sigma)
    y = exp(-(x-miu).^2 / (2*sigma^2)) ./ (sigma*(2*pi)^0.5);
end