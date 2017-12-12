clear
load train.mat
load test.mat
load param.mat

X = normalizeFea(X);
Y = normalizeFea(Y);
X_te = normalizeFea(X_te);

Y_pred = predict(X_te, W, b);

[n_tr,~] = size(X);
[n_te,~] = size(X_te);

distY = zeros(n_te, n_tr);
for i = 1 : n_te
    for j = 1: n_tr
        tmp = bsxfun(@minus,Y_pred(i,:), Y);
        tmp = sum( abs(tmp),2);
        distY(i,:) = tmp';
    end
end

distX = zeros(n_te, n_tr);
for i = 1 : n_te
    for j = 1: n_tr
        tmp = bsxfun(@minus,X_te(i,:), X);
        tmp = sum( abs(tmp),2);
        distX(i,:) = tmp';
    end
end

lambda = 0.4;

dist = lambda * distX + (1-lambda) *distY;
for i = 1 : n_te
    [value, rank] = sort(dist(i,:),'ascend');
    disp(['test_id: ', num2str(i-1)]);
    disp(['similar id: ', num2str(rank(1:3)-1)]);
end




