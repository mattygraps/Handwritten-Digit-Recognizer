function Y_pred = predict(X, parameters)

    N = size(X,2);                                  % gets number of samples
    activations = forward_propagation(X,parameters);% runs forward propogation
    final_layer_output = activations{end};          % finds the output of neural network
    Y_pred = zeros(10,N);                           % creates array for predictions

    for i = 1:N
        [M,I] = max(final_layer_output(:, i));      % finds highest probabilty
        Y_pred(I, i) = 1;                           % labels number as the highest probability
    end 
    
end
