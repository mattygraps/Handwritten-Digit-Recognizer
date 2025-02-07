function parameters = update_parameters(parameters, gradients, learning_rate)

    L = length(gradients);
    
    
    for i = 1:L
        parameters{i}.W = parameters{i}.W - learning_rate*gradients{i}.dW;
        parameters{i}.b = parameters{i}.b - learning_rate*gradients{i}.db;
    end

end