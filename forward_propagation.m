

function activations = forward_propagation(X, parameters)
    % initialize cells and variables
    L = length(parameters);
    A = X;
    activations = cell(1,L+1);
    activations{1} = A;

    % iterate through param. to find activations at each layer
    for i = 1:L
        Z = parameters{i}.W * activations{i} + parameters{i}.b;

        if i == L
            A = softmax(Z);
        else
            A = tanh2(Z);
        end 
        activations{i+1} = A;

    end

end