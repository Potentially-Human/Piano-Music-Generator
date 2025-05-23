import torch

def train_step(x, y, model, optimizer, loss_function):
    model.train()
    optimizer.zero_grad()
    pred_y = model(x)
    loss = loss_function(y, pred_y)

    loss.backward()
    optimizer.step()

    return loss.item()