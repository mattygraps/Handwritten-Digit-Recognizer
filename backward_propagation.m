

function gradients = backward_propagation(X, Y, parameters, activations)

    % initialize cells and array
    L = length(parameters);
    m = size(X,2);
    gradients = cell(1,L);
    dZ = activations{end} - Y;
    % define last struct of gradient values
    gradients{L}.dW = dZ*activations{L}'/m;
    gradients{L}.db = sum(dZ,2)/m;

    % iterate from the last hidden layer down to first hidden layer
    for i = (L-1):-1:1
        dA = parameters{i + 1}.W'*dZ;
        dZ = dA.*(1-tanh2(activations{i+1}).^2);

        if i==1
            A_prev = X;
        else
            A_prev = activations{i};
        end
        % define stuct(hidden layer number) within gradient
        gradients{i}.dW = dZ*A_prev'/m;
        gradients{i}.db = sum(dZ,2)/m;

    end

end 








