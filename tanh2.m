function Z = tanh2(X)
    % applied between outputs of hidden layer
    Z = 2./(1+exp(-2*X)) - 1;

end
