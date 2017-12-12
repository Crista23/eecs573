function [W,b] = train_lr(X, Y, L, alpha, beta, rank, learning_rate, epoch)
    rng(1);
    [n_inst, n_fea] = size(Y);
    U = 0.01 * randn(n_fea, rank);
	V = 0.01 * randn(n_fea, rank);
    b = 0.01 * randn(n_fea, 1);
    XT = X';
    XTLX = XT*L*X;
    XTX = X'*X;
    XTY = X'*Y;
	YTX = Y'*X;
    l = ones(n_inst,1);
    E = eye(n_fea,n_fea);
    loss = [];
    loss(1) = 1/n_inst * norm(Y-X*U*V'-l*b', 'fro')^2 + ...
                alpha*trace((X*U*V'+l*b')'*L*(X*U*V'+l*b'))...
                +beta*(norm(U,'fro')^2+norm(V,'fro')^2+norm(b)^2);
    for i = 2 : epoch
        gradU = -2/n_inst*XTY*V + 2*(1/n_inst*XTX + alpha*XTLX)*U*V'*V ...
            + 2*(1/n_inst*XT + alpha*XT*L)*l*b'*V + 2*beta*U;
        U = U - learning_rate * gradU;
        
        gradV = -2/n_inst*YTX*U + 2*V*U'*(1/n_inst*XTX + alpha*XTLX)*U ...
            + 2*b*l'*(1/n_inst*X + alpha*L*X)*U + 2*beta*V;
        V = V - learning_rate * gradV;
        
        gradb = -2/n_inst*Y'*l + 2*V*U'*(1/n_inst*XT + alpha*XT*L)*l ...
            + 2*(1/n_inst*(l')*l + alpha*l'*L*l)*b + 2*beta*b;
        b = b - learning_rate * gradb;
        
        loss(i) = 1/n_inst * norm(Y-X*U*V'-l*b', 'fro')^2 + ...
                    alpha*trace((X*U*V'+l*b')'*L*(X*U*V'+l*b'))...
                    + beta*(norm(U,'fro')^2+norm(V,'fro')^2+norm(b)^2);
    end
    %figure
    %plot(1:epoch, loss);
    W = U*V';
end