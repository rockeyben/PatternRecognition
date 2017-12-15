clear all;
clc;
data = importdata('swiss-data.txt');
label = data(:,1);
x = data(:, 2:4);

%PCA
cov_x = cov(x);
[V, D] = eig(cov_x);
d = diag(D);
[sort_d, idx] = sort(d,'descend');
a1 = V(:,idx(1));
a2 = V(:,idx(2));
A = [a1,a2];
PCA_res = x * A;
size(PCA_res)
pc1 = PCA_res(:,1);
pc2 = PCA_res(:,2);
figure;
scatter(pc1,pc2,'.'); 





