import argparse
import sys

import torch
import click
from torch import nn, optim
import torch.nn.functional as F

from data import mnist
from model import MyAwesomeModel

@click.group()
def cli():
    pass


@click.command()
@click.option("--lr", default=1e-3, help='learning rate to use for training')
def train(lr):
    print("Training day and night")
    # parser = argparse.ArgumentParser(description='Training arguments')
    # parser.add_argument('--lr', default=1e-3)
    # # add any additional argument that you want
    # args = parser.parse_args(sys.argv[2:])
    # print(args)
    
    # TODO: Implement training loop here
    model = MyAwesomeModel()
    #model = model.to(device)
    train_set = mnist(train=True)
    dataloader = torch.utils.data.DataLoader(train_set, batch_size=128)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    
    n_epoch = 2
    for epoch in range(n_epoch):
        print(epoch)
        loss_tracker = []
        for batch in dataloader:
            optimizer.zero_grad()
            x, y = batch
            preds = model(x)#.to(self.device))
            loss = criterion(preds, y)#.to(self.device))
            loss.backward()
            optimizer.step()
            loss_tracker.append(loss.item())
        print(f"Epoch {epoch+1}/{n_epoch}. Loss: {loss}")        
    torch.save(model.state_dict(), 'trained_model.pt')

#train()

@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    state_dict = torch.load(model_checkpoint)
    model=MyAwesomeModel()
    model.load_state_dict(state_dict)
    test_set = mnist(train=False)
    dataloader = torch.utils.data.DataLoader(test_set, batch_size=128)
    
    correct, total = 0, 0
    for batch in dataloader:
        x, y = batch
        
        preds = model(x)#.to(self.device))
        preds = preds.argmax(dim=-1)
        
        correct += (preds == y).sum().item()#.to(self.device)).sum().item()
        total += y.numel()
        
    print(f"Test set accuracy {correct/total}")

cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()
    #train()
    #evaluate()