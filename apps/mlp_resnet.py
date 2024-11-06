import sys

sys.path.append("../python")
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)
# MY_DEVICE = ndl.backend_selection.cuda()


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    block = nn.Sequential(
        nn.Linear(dim, hidden_dim),
        norm(hidden_dim), 
        nn.ReLU(),
        nn.Dropout(drop_prob),
        nn.Linear(hidden_dim, dim),
        norm(dim)
    )
    block = nn.Residual(block)
    block = nn.Sequential(
        block,
        nn.ReLU()
    )
    return block
    ### END YOUR SOLUTION


def MLPResNet(dim, hidden_dim=100, num_blocks=3, num_classes=10, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    res_net = nn.Sequential(
        nn.Linear(dim, hidden_dim),
        nn.ReLU(),
        *[ResidualBlock(hidden_dim, hidden_dim // 2, norm, drop_prob) for _ in range(num_blocks)],
        nn.Linear(hidden_dim, num_classes)
    )
    return res_net
    ### END YOUR SOLUTION


def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    tot_loss = []
    error_rate = 0.0
    loss_fn = nn.SoftmaxLoss()
    if opt is None:
        model.eval()
        for batch_x, batch_y in dataloader:
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            tot_loss.append(loss.numpy())
            error_rate += np.sum(logits.numpy().argmax(axis=1) != batch_y.numpy())    
    else:
        model.train()
        for batch_x, batch_y in dataloader:
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            tot_loss.append(loss.numpy())
            error_rate += np.sum(logits.numpy().argmax(axis=1) != batch_y.numpy())
            opt.reset_grad()
            loss.backward()
            opt.step()
    data_len = len(dataloader.dataset)
    return error_rate / data_len, np.mean(tot_loss)

    ### END YOUR SOLUTION



def train_mnist(batch_size=100, epochs=10, optimizer=ndl.optim.Adam,
                lr=0.001, weight_decay=0.001, hidden_dim=100, data_dir="data"):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    train_dataset = ndl.data.MNISTDataset(f'{data_dir}/train-images-idx3-ubyte.gz', f'{data_dir}/train-labels-idx1-ubyte.gz')
    test_dataset = ndl.data.MNISTDataset(f'{data_dir}/t10k-images-idx3-ubyte.gz', f'{data_dir}/t10k-labels-idx1-ubyte.gz')
    train_dataloader = ndl.data.DataLoader(train_dataset, batch_size, True)
    test_dataloader = ndl.data.DataLoader(test_dataset, batch_size, False)

    model = MLPResNet(dim=28 * 28, hidden_dim=hidden_dim)
    opt = optimizer(params=model.parameters(), lr=lr, weight_decay=weight_decay)
    
    for i in range(epochs):
        train_error_rate, train_mean_loss = epoch(train_dataloader, model, opt)
        test_error_rate, test_mean_loss = epoch(test_dataloader, model, None)
    
    return (train_error_rate, train_mean_loss, test_error_rate, test_mean_loss)
    ### END YOUR SOLUTION


if __name__ == "__main__":
    train_mnist(data_dir="../data")
