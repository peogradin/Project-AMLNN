
import torch
import deeplay as dl
import matplotlib.pyplot as plt

def plot_training(epochs, train_losses, val_losses, benchmark):
    """Plot the training and validation losses."""
    plt.plot(range(epochs), train_losses, label="Training Loss")
    plt.plot(range(epochs), val_losses, "--", label="Validation Loss")
    plt.plot([0, epochs - 1], [benchmark, benchmark], ":k", label="Benchmark")
    plt.xlabel("Epoch")
    plt.xlim([0, epochs - 1])
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    

def train_model(model, epochs, optimizer, train_loader, val_loader, benchmark, loss=torch.nn.L1Loss()):
    '''Create the regressor for the model including optimizer and loss. Trains the model and plots the training and validation loss'''
    model_reg = dl.Regressor(model, optimizer=optimizer, loss=loss).create()

    trainer = dl.Trainer(max_epochs=epochs, accelerator="auto")
    trainer.fit(model_reg, train_loader, val_loader)

    train_losses = trainer.history.history["train_loss_epoch"]["value"]
    val_losses = trainer.history.history["val_loss_epoch"]["value"][1:]

    plot_training(epochs, train_losses, val_losses, benchmark)
    #torch.save(model.state_dict(), "trained_model.pth")
    #return "trained_model.pth"

def get_device():
    """Select device where to perform the computations."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
    
def test_model_output(model, data_loader, num_tests=20):
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            x_batch, y_batch = batch
            #print(y_batch)
            test_prediction = model(x_batch)
            
            break

    print("Last 10 timesteps (input) and true vs predicted temperature:")
    for i in range(num_tests):
        
        last_input_temp = x_batch[i, -1, 1].item()  
        true_temp = y_batch[i].item()
        predicted_temp = test_prediction[i].item()

        print(f"Sample {i+1}: Last input temp = {last_input_temp:.2f}, True = {true_temp:.2f}, Pred = {predicted_temp:.2f}")

import numpy as np
