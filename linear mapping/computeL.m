function L = computeL(k, Y)
    %compute the similarity between each row
    %return the associated laplacian matrix L
    [n_inst,~] = size(Y);
    S = zeros(n_inst, n_inst);
    for i = 1 : n_inst
        tmp = bsxfun(@minus,Y(i,:),Y);
        tmp = sum(tmp.*tmp,2);
        S(i,:) = tmp;
    end
    sigma = max(max(S));
    S = S/sigma;
    S = exp(-S);
    S = S - eye(n_inst, n_inst); 
    S_new = zeros(size(S));
    for i = 1: n_inst
        [~,idx] = sort(S(i,:),'descend');
        S_new(i,idx(1:k)) = S(i,idx(1:k));
    end
    S_new = (S_new + S_new') / 2;
    A = diag(sum(S_new,1).^(-0.5));
    L = eye(n_inst, n_inst) - A*S_new*A;
end