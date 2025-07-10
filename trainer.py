import torch

def train_step(x, y, model, optimizer, loss_function, max_offset, max_duration, max_volume, device):
    model.train()
    optimizer.zero_grad()
    pred_y = model(x)
    loss = loss_function(y, pred_y, max_offset, max_duration, max_volume, device)

    loss.backward()
    optimizer.step()

    return loss.item()