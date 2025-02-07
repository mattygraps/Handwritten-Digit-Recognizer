function visualize_history(epochs, train, testAccuracy, learning_rate, numLayer)

    epochs_array = linspace(1,epochs,epochs);

    fig = figure(1);
    
    % plot training loss
    subplot(1,2,1);
    plot(epochs_array,train);
    xlabel('Epochs');
    ylabel('Training Loss');

    % plot testing accuracy
    subplot(1,2,2);
    plot(epochs_array,testAccuracy);
    xlabel('Epochs');
    ylabel('Testing Accuracy');



    % adds common title for figure
    titleText = sprintf('Epochs: %i; Learning Rate: %.4f; Number of Hidden Layers: %i',epochs,learning_rate,numLayer);
    titleFontSize = 12;
    titleFontWeight = 'bold';

    % adjust position of title
    sgtitle(titleText, 'FontSize', titleFontSize, 'FontWeight', titleFontWeight);

    % save figure
    saveTitle = sprintf('model_%.4f_%i_%i.png', learning_rate, numLayer, epochs);
    saveas(fig, saveTitle);

end