clear all;
% SVM with RBF kernel
% import data
raw_data = importdata('hw4_data.txt');
x = raw_data(:, 1:3);
y = raw_data(:, 4);
y(y==0)=-1;
[n, d] = size(x);

% some hyper-params
threshold = 0.1;
C = 20;
sigma = 20;

% SMO algorithm
a = ones(1, n);
newW=0;
while true
    i = randi([1,n]);
    j = randi([1,n]);
    while i==j
        j = randi([1,n])
    end
    a2new = a(j)+y(j)*(E(a,y,x,i,n,sigma)-E(a,y,x,j,n,sigma))/(K(x(i,:),x(i,:),sigma)+K(x(j,:),x(j,:),sigma)-2*K(x(i,:),x(j,:),sigma));
    U = 0;
    V = C;
    if y(i)~=y(j)
        U = max([0, a(j)-a(i)]);
        V = min([C, C-a(i)+a(j)]);
    end
    if y(i)==y(j)
        U = max([0, a(i)+a(j)-C]);
        V = min([C, a(i)+a(j)]);
    end
    if a2new > V
        a2new = V;
    end
    if a2new < U
        a2new = U;
    end
    
    a1new = a(i)+y(i)*y(j)*(a(j)-a2new);
    
    a(i)=a1new;
    a(j)=a2new;
    
    newW = W(a,x,y,n,sigma)
    
    sum_kernel = 0;
    for k=1:n
        sum_kernel = sum_kernel + a(k)*y(k)*K(x(1,:),x(k,:),sigma);
    end
    b = (1-y(1)*sum_kernel)/y(1);
    ksi = zeros(1,n);
    for k=1:n
        sum_kernel = 0;
        for t=1:n
            sum_kernel = sum_kernel + a(t)*y(t)*K(x(k,:),x(t,:),sigma);
        end
        ksi(k)=max([0, 1-y(k)*(sum_kernel+b)]);
    end
    
    judge = sum(a)-newW+C*sum(ksi);
    
    if (judge-newW)/(judge+1) < threshold
        break;
    end     
end
sum_kernel = 0;
for k=1:n
    sum_kernel = sum_kernel + a(k)*y(k)*K(x(1,:),x(k,:),sigma);
end
b = (1-y(1)*sum_kernel)/y(1);

x1 = x(:,1);
x2 = x(:,2);
x3 = x(:,3);
figure
scatter3(x1(y==1),x2(y==1),x3(y==1),'b');
hold on;
scatter3(x1(y==-1),x2(y==-1),x3(y==-1),'r');
hold on;

pred = zeros(1,n);
for i=1:n
    x_sample = x(i,:);
    sum_kernel=0;
    for j=1:n;
        sum_kernel = sum_kernel + a(j)*y(j)*K(x_sample, x(j,:), sigma);
    end
    pred(i) = sum_kernel+b;
end
scatter3(x1(pred>=0),x2(pred>=0),x3(pred>=0),'b+');
hold on;
scatter3(x1(pred<0),x2(pred<0),x3(pred<0),'r*');
title(['final W(a): ',num2str(newW),' \sigma=',num2str(sigma), ' C=',num2str(C),' threshold=',num2str(threshold)])
legend('original positive data', 'original negative data', 'predicted positive data', 'predicted negative data');

function res = W(a, x, y, n, sigma)
    res = sum(a);
    for i=1:n
        for j=1:n
            res = res + 0.5*a(i)*a(j)*y(i)*y(j)*K(x(i,:), x(j,:), sigma);
        end
    end
end

function res = E(a, y, x, i, n, sigma)
    res = 0;
    for j=1:n
        res = res + a(j)*y(j)*K(x(i,:), x(j,:), sigma);
    end
    res = res - y(i);
end

function res = K(xi, xj, sigma)
    res = exp(-0.5*sum((xi-xj).^2)/sigma^2);
end

