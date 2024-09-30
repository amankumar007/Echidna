import torch
import torch.nn as nn


def train():

    model = nn.Linear(4,2)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    for epoch in range(10):
        # Converting inputs and labels to Variable
        inputs = torch.Tensor([0.8,0.4,0.4,0.2])
        labels = torch.Tensor([1,0])

        # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward
        optimizer.zero_grad()

        # get output from the model, given the inputs
        outputs = model(inputs)

        # get loss for the predicted output
        loss = criterion(outputs, labels)
        print(loss)

        # get gradients w.r.t to parameters
        loss.backward()

        # update parameters
        optimizer.step()

        print('epoch {}, loss {}'.format(epoch, loss.item()))

if __name__ == "__main__":
    train()