%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Final Project
% Name: Matthew Gallati
% UID: 005767973
%{
Description:
This script trains a feedforward neural network to classify handwritten
number digits 0-10. It does this by flattening the image from a 28x28 pixel
image to a 784x1 pixel image than passing it through a multi-layer neural
network which returns a 10x1 output vector. The output vector consists of
nine, 0 elements and one, 1 element which represents the number prediction
of the network. 
%}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all; close all; clc;

%% Load Data
[X_train, Y_train, X_test, Y_test] = load_train_and_test_data();

%% Define Archetecture and Hyperparameters
input_size = size(X_train, 1); % input size of the FNN
output_size = size(Y_train, 1);% output size (number of classes) of the FNN
neurons = 64; % neurons each layer
numLayer = 2;
lr = 0.01; % learning rate
epochs = 150; % epochs
layer_dims = zeros(1, numLayer + 2);
layer_dims(1) = input_size;
layer_dims(end) = output_size;

for i = 1:numLayer
    layer_dims(i+1) = neurons;
end

%% Train Model
parameters = initialize_parameters(layer_dims);
fprintf('Initialize cost somewhere so that we do not have warnings.\n');
% Train the model using mini-batch gradient descent
m = size(X_train, 2);
batch_size = 64;
num_batches = floor(m / batch_size);
trainLoss = zeros(epochs, 1);
testAccuracy = zeros(epochs, 1);

for i = 1:epochs

    % Shuffle the training data
    indices = randperm(m);
    X_train = X_train(:, indices);
    Y_train = Y_train(:, indices);
    % Initialize cost matrix
    cost = zeros(num_batches, batch_size);
    % Train on mini-batches

    for j = 1:num_batches
        % normalize X and Y for input into neural network
        X_batch = X_train(:, (j-1)*batch_size+1:j*batch_size);
        Y_batch = Y_train(:, (j-1)*batch_size+1:j*batch_size);
        % run a forward pass
        forward_pass = forward_propagation(X_batch, parameters);
        % calculate error
        cost(j,:) = compute_cost(forward_pass{end}, Y_batch);
        % run back propogation to get gradient
        gradients = backward_propagation(X_batch, Y_batch, parameters, forward_pass);
        % use gradient to update parameters
        parameters = update_parameters(parameters, gradients, lr);
    end
    
    % predict test cases
    Y_pred = predict(X_test, parameters);
    % calculate accuracy of test predictions
    acc = accuracy(Y_pred, Y_test);
    % Print the cost each epoch
    fprintf('Loss after epoch %d: Training: %f\n', i, norm(cost));
    % add to accuracy and loss arrays 
    trainLoss(i) = norm(cost);
    testAccuracy(i) = acc;
end

%% Evaluate Model w/ Test Set
fprintf('Test accuracy: %f\n', testAccuracy(end));

%% Visualize Training Process
% plot epochs vs. trainLoss and testAccuracy
visualize_history(epochs, trainLoss, testAccuracy, lr, numLayer);











