load train.mat
X = normalizeFea(X);
Y = normalizeFea(Y);
[n_inst,n_fea] = size(Y);
k = 10;
alpha = 0.0001;
beta = 0.001;
learning_rate = 0.01;
epoch = 200;
rand_folds = 3;
MAE = [];
RMSE = [];
%for r = 1 : 20
for r = [12,14,16,18,20]
    for i = 1:rand_folds
        [tr_split, val_split] = tr_te_split(n_inst, rand_folds, 2);
        X_tr  = X(tr_split(i,:),:);
        Y_tr  = Y(tr_split(i,:),:);
        X_val = X(val_split(i,:),:);
        Y_val = Y(val_split(i,:),:);
        L = computeL(k, Y_tr);

        % low rank algorithm
        rank = round(0.05*r*n_fea);
        [W,b] = train_lr(X_tr, Y_tr, L, alpha, beta, rank, learning_rate, epoch);


        %predict
        Y_pred = predict(X_val, W, b);
        Y_error = Y_val-Y_pred;
        Y_error(Y_error>1) = 1;
        Y_error(Y_error<0) = 0;
        MAE(i) = sum(mean(abs(Y_error)));
        MSE(i) = sum(mean(Y_error .* Y_error));
    end
disp(['rank: ', num2str(r)])
disp(['MAE: ',num2str(mean(MAE))])
disp(['MSE: ',num2str(mean(MSE))])
end
%save('param.mat','W','b')
