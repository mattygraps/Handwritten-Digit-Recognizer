function predict_single_image(parameters)
    
    % Visualizes the input image and plots a bar graph of probabilities for
    % each class
    % Inputs:
    % parameters: a struct containing the weights and biases (W1, b1,
    % W2, b2, etc.)
    % Output:
    % (None)
    % Load data

    load('test_images.mat');
    index = randi(length(pixel));
    X = reshape( pixel(:,:,index), [size(pixel, 1)*size(pixel, 2), 1]) / 255;
    forward = forward_propagation(X, parameters);
    probabilities = forward{end};
    
    fig1 = figure(2);
    subplot(1, 2, 1)
    imshow(pixel(:,:,index))
    title('Input Image')
    subplot(1, 2, 2)
    bar(0:9, probabilities)
    xlabel('Classes');
    ylabel('Probability');
    title('Probability Distribution')
    saveas(fig1, 'predict_single_image.png');

end