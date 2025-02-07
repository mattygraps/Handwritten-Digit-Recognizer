function acc = accuracy(Y_pred, Y)

    % get size of Y_predictions
    [K,N] = size(Y_pred);
    
    % create an empty array to store accuracy for each set of predictions
    accuracy_array = zeros(N,1);
    
    % iterate to number of predictions
    for i = 1:N
        pred_i = Y(:,i);
        act_i = Y_pred(:,i);

        if all(pred_i == act_i) == 1
            accuracy_array(i) = 1;
        end
    end
    % accuracy is the sum of accurate prediction/ total number of
    % predictions
    acc = sum(accuracy_array)/N;

end