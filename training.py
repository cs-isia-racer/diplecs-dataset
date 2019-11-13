def train(model, dataloader, criterion, optimizer, device, epochs=5):
    """trains the given model using the given crierion and optimizer

    :model: a torch model to train
    :dataloader: the dataloader to use
    :criterion: the criterion to use (for instance nn.MSELoss)
    :optimizer: optimizer to use (for instance optim.SGD)
    :epochs: the number of epochs to use
    :device: the device to use
    :returns: the trained model

    """
    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)

        running_loss = 0.0

        for batch in dataloader:
            inputs, outs = batch['image'].to(device), batch['feats'].to(device)

            # zero the gradients
            optimizer.zero_grad()

            # forward
            outputs = model(inputs)
            loss = criterion(outputs, outs)

            # backward
            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            print('.', end='')

        epoch_loss = running_loss / len(dataloader)
        print('\nLoss: {:.4f}'.format(epoch_loss))

    return model
