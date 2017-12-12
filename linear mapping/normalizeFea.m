function X_new = normalizeFea(X)
    min_fea = min(X,[], 1);
    max_fea = max(X,[], 1);
    min_fea = repmat(min_fea, [size(X,1),1]);
	max_fea = repmat(max_fea, [size(X,1),1]);
    X_new = (X - min_fea) ./ (max_fea - min_fea);
end