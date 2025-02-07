



function [X_train, Y_train, X_test, Y_test] = load_train_and_test_data()
    
    %% Define Training Arrays
    % load data
    load('train_images.mat');
    load('train_labels.mat');
    % define X_train
    [L, W, numSamples] = size(pixel);
    pixel = reshape(pixel, [L*W, numSamples]);
    X_train = pixel/255;
    
    % define Y_train
    Y_train = zeros(1, numSamples);
    for k = 1:numSamples
        Y_train(label(k) + 1, k) = 1;
    end
    
    %% Define Testing Arrays
    % load data
    load('test_images.mat');
    load('test_labels.mat');
    % define X_train
    [L, W, numSamples] = size(pixel);
    pixel = reshape(pixel, [L*W, numSamples]);
    X_test = pixel/255;
    
    % define Y_train
    Y_test = zeros(1, numSamples);
    for k = 1:numSamples
        Y_test(label(k) + 1, k) = 1;
    end

end




