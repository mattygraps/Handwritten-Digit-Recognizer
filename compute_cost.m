function cost = compute_cost(AL, Y)
    
    % size of forward propogation output is taken
    [K,N] = size(AL);
    % empty array to hold cost values is initialized
    cost_test = zeros(1,N);

    % for loop iterates from one to column size
    for i = 1:N    
        % calculates cross-entropy loss
        cost_test(i) = -sum(Y(:,i).*log(AL(:,i)));     
    end

    % defines output
    cost = cost_test;
    
end