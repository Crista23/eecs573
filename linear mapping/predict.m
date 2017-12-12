function Y_pred = predict(X_te, W, b)
    n_inst = size(X_te,1);
    l = ones(n_inst,1);
    Y_pred = X_te * W + l*b';
end