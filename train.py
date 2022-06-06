import numpy as np

from pinn_model import *
import time
import pandas as pd
import os

# part time data training (interpolation & extrapolation) - no data normalization
# training code body
portion = 0.5
x, y, u, v, p, feature_mat = read_data_portion(filename_data, portion)
layer_mat = [2, 20, 20, 20, 20, 20, 20, 20, 20, 2]
X_random = shuffle_data(x, y, u, v, p)
view_x = X_random.data.numpy()
# Create a PINN model instance and assign the instance to the corresponding device
pinn_net = PINN_Net(layer_mat)
pinn_net = pinn_net.to(device)
# Loss functions and optimizers
mse = torch.nn.MSELoss()
# A list to record the loss of each part
losses = np.empty((0, 3), dtype=float)

if os.path.exists(filename_save_model):
    pinn_net.load_state_dict(torch.load(filename_load_model, map_location=device))
if os.path.exists(filename_loss):
    loss_read = pd.read_csv('loss_0.005_12.0.csv', header=None)
    losses = loss_read.values
# Optimizer and Learning Rate Decay Settings
optimizer = torch.optim.Adam(pinn_net.parameters())
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.2)
epochs = 500

# Select batch size data_loader can also be used here
batch_size = 25
inner_iter = int(X_random.size(0) / batch_size)

for epoch in range(epochs):
    for batch_iter in range(inner_iter):
        optimizer.zero_grad()
        # Randomly take batches in the full set
        x_train = X_random[batch_iter*batch_size:((batch_iter+1)*batch_size), 0].view(batch_size, 1)
        y_train = X_random[batch_iter*batch_size:((batch_iter+1)*batch_size), 1].view(batch_size, 1)
        # t_train = X_random[batch_iter*batch_size:((batch_iter+1)*batch_size), 2].view(batch_size, 1)
        u_train = X_random[batch_iter*batch_size:((batch_iter+1)*batch_size), 2].view(batch_size, 1)
        v_train = X_random[batch_iter*batch_size:((batch_iter+1)*batch_size), 3].view(batch_size, 1)
        p_train = X_random[batch_iter*batch_size:((batch_iter+1)*batch_size), 4].view(batch_size, 1)

        # MSE set at zeros for computing the error of differential equations
        zeros = np.zeros((batch_size, 1))
        # Clone the batch from the full set
        batch_t_x = x_train.clone().requires_grad_(True).to(device)
        batch_t_y = y_train.clone().requires_grad_(True).to(device)
        # batch_t_t = t_train.clone().requires_grad_(True).to(device)
        batch_t_u = u_train.clone().requires_grad_(True).to(device)
        batch_t_v = v_train.clone().requires_grad_(True).to(device)
        batch_t_p = p_train.clone().requires_grad_(True).to(device)
        batch_t_zeros = torch.from_numpy(zeros).float().requires_grad_(True).to(device)
        # Delete unnecessary memory space


        # Call the f_equation function to calculate the loss function
        u_predict, v_predict, f_equation_x, f_equation_y = f_equation_identification(batch_t_x, batch_t_y,
                                                                                         pinn_net, lam1=1.0, lam2=0.01)

        # Calculate the loss function
        mse_predict = mse(u_predict, batch_t_u) + mse(v_predict, batch_t_v)
        mse_equation = mse(f_equation_x, batch_t_zeros) + mse(f_equation_y, batch_t_zeros)
        loss = mse_predict + mse_equation
        loss.backward()
        del x_train, y_train, u_train, v_train, p_train, zeros
        optimizer.step()
        with torch.autograd.no_grad():
            # Output status every 200 iterations
            if (batch_iter + 1) % 50 == 0:
                # add loss to losses
                loss_all = loss.cpu().data.numpy().reshape(1, 1)
                loss_predict = mse_predict.cpu().data.numpy().reshape(1, 1)
                loss_equation = mse_equation.cpu().data.numpy().reshape(1, 1)
                loss_set = np.concatenate((loss_all, loss_predict, loss_equation), 1)
                losses = np.append(losses, loss_set, 0)
                print("Epoch:", (epoch+1), "  Bacth_iter:", batch_iter + 1, " Training Loss:", round(float(loss.data), 8))
            # Save the state every 1 epoch (model state, loss, number of iterations)
            if (batch_iter + 1) % inner_iter == 0:
                torch.save(pinn_net.state_dict(), filename_save_model)
                loss_save = pd.DataFrame(losses)
                loss_save.to_csv(filename_loss, index=False, header=False)
                del loss_save
    scheduler.step()
print("one oK")
torch.save(pinn_net.state_dict(), filename_save_model)
