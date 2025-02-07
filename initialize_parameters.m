


function parameters = initialize_parameters(layer_dims)
    
    % establish L size and parameter cell size
    L = length(layer_dims);
    parameters = cell(1, L-1);

    % sets first hidden layers weights and biases
    parameters{1}.W = randn(layer_dims(2), layer_dims(1));
    parameters{1}.b = zeros(layer_dims(2), 1);

    % fills out wieghts and biases more for rest of hidden/output layersTh
    for i = 2:L-1
        parameters{i}.W = randn(layer_dims(i+1), layer_dims(i));
        parameters{i}.b = zeros(layer_dims(i+1), 1);
    end
end

