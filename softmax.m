function Z = softmax(X)
    % applied between outputs of final hidden layer and output layer
    Z = exp(X)./sum(exp(X));

end
