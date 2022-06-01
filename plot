# Visually compare predicted and actual results
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from pinn_model import *
from scipy.io import savemat


filename_load_model = 'NS_model_train_0.02_8.0.pt'
filename_data = './CFD_0.02_8.0.mat'
x, y, u, v, p, N, feature_mat = read_data(filename_data)
data_stack = np.concatenate((x, y, u, v, p), axis=1)
del x, y, u, v, p
layer_mat = [2, 20, 20, 20, 20, 20, 20, 20, 20, 2]
pinn_net = PINN_Net(layer_mat)
pinn_net.load_state_dict(torch.load(filename_load_model, map_location=device))


# Comparison of selected moments
def compare_at_select_time(data_stack, pinn_example):
    x = data_stack[:, 0].copy().reshape(-1, 1)
    y = data_stack[:, 1].copy().reshape(-1, 1)
    u = data_stack[:, 2].copy().reshape(-1, 1)
    v = data_stack[:, 3].copy().reshape(-1, 1)
    p = data_stack[:, 4].copy().reshape(-1, 1)
    min_data = np.nanmin(data_stack, 0).reshape(1, data_stack.shape[1])
    max_data = np.nanmax(data_stack, 0).reshape(1, data_stack.shape[1])
    # Preserve unique coordinates in the dataset
    x = np.unique(x).reshape(-1, 1)
    y = np.unique(y).reshape(-1, 1)
    # Select u,v,p at the specified time
    u_selected = u
    v_selected = v
    p_selected = p
    # give the grid points of x,y
    mesh_x, mesh_y = np.meshgrid(x, y)
    x_flatten = np.ndarray.flatten(mesh_x).reshape(-1, 1)
    y_flatten = np.ndarray.flatten(mesh_y).reshape(-1, 1)
    # t_flatten = np.ones((x_flatten.shape[0], 1)) * select_time

    x_selected = torch.tensor(x_flatten, requires_grad=True, dtype=torch.float32).to(device)
    y_selected = torch.tensor(y_flatten, requires_grad=True, dtype=torch.float32).to(device)
    # t_selected = torch.tensor(t_flatten, requires_grad=True, dtype=torch.float32).to(device)
    del x_flatten, y_flatten
    u_predict, v_predict, p_predict, f_equation_x, f_equation_y = f_equation_inverse(x_selected, y_selected,
                                                                                     pinn_example)
    # draw
    u_predict = u_predict.data.numpy().reshape(mesh_x.shape)
    v_predict = v_predict.data.numpy().reshape(mesh_x.shape)
    p_predict = p_predict.data.numpy().reshape(mesh_x.shape)
    u_selected = np.transpose(u_selected.reshape(mesh_x.shape))
    v_selected = np.transpose(v_selected.reshape(mesh_x.shape))
    p_selected = np.transpose(p_selected.reshape(mesh_x.shape))

    savemat("u_predict_0.02_0.3.mat.mat", {'u_predict': u_predict})
    savemat("v_predict_0.02_0.3.mat.mat", {'v_predict': v_predict})
    savemat("p_predict_0.02_0.3.mat.mat", {'p_predict': p_predict})

    plot_compare(u_selected, u_predict, name='u', min_value=min_data[0, 2], max_value=max_data[0, 2])
    plot_compare(v_selected, v_predict, name='v', min_value=min_data[0, 3], max_value=max_data[0, 3])
    plot_compare(p_selected, p_predict, name='p', min_value=min_data[0, 4], max_value=max_data[0, 4])

    print('ok')


# Comparison of selected moments
def compare_at_select_time_simple_norm(data_stack, feature_mat, pinn_example):
    x = data_stack[:, 0].copy().reshape(-1, 1)
    y = data_stack[:, 1].copy().reshape(-1, 1)
    u = data_stack[:, 2].copy().reshape(-1, 1)
    v = data_stack[:, 3].copy().reshape(-1, 1)
    p = data_stack[:, 4].copy().reshape(-1, 1)
    min_data = np.min(data_stack, 0).reshape(1, data_stack.shape[1])
    max_data = np.max(data_stack, 0).reshape(1, data_stack.shape[1])
    # Preserve unique coordinates in the dataset
    x = np.unique(x).reshape(-1, 1)
    y = np.unique(y).reshape(-1, 1)
    # Select u,v,p 
    u_selected = u
    v_selected = v
    p_selected = p
    # give the grid points for x,y
    mesh_x, mesh_y = np.meshgrid(x, y)
   

    x_selected = torch.tensor(x, requires_grad=True, dtype=torch.float32).to(device)
    y_selected = torch.tensor(y, requires_grad=True, dtype=torch.float32).to(device)
    u_predict, v_predict, p_predict,f_equation_c, f_equation_x, f_equation_y = f_equation_inverse_simple_norm(x_selected, y_selected, feature_mat, pinn_example)
    u_predict = u_predict * feature_mat[0, 3]
    v_predict = v_predict * feature_mat[0, 4]
    p_predict = p_predict * feature_mat[0, 5]
    # draw
    u_predict = u_predict.data.numpy().reshape(mesh_x.shape)
    v_predict = v_predict.data.numpy().reshape(mesh_x.shape)
    p_predict = p_predict.data.numpy().reshape(mesh_x.shape)
    u_selected = u_selected.reshape(mesh_x.shape)
    v_selected = v_selected.reshape(mesh_x.shape)
    p_selected = p_selected.reshape(mesh_x.shape)
    plot_compare(u_selected, u_predict, name='u', min_value=min_data[0, 3], max_value=max_data[0, 3])
    plot_compare(v_selected, v_predict, name='v', min_value=min_data[0, 4], max_value=max_data[0, 4])
    plot_compare(p_selected, p_predict, name='p', min_value=min_data[0, 5], max_value=max_data[0, 5])
    print('ok')


def plot_compare(q_selected, q_predict, min_value, max_value, name='q'):
    fig_q = plt.figure(figsize=(10, 4))
    v_norm = mpl.colors.Normalize(vmin=min_value, vmax=max_value)
    plt.subplot(1, 2, 1)
    plt.imshow(q_selected, cmap='jet', norm=v_norm)
    plt.title("True_value:" + name + "(x,y)")
    plt.ylabel('Y')
    plt.xlabel('X')
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(q_predict, cmap='jet', norm=v_norm)
    plt.title("Predict_value:" + name + "(x,y)")
    plt.ylabel('Y')
    plt.xlabel('X')
    plt.colorbar()
    plt.show()


compare_at_select_time(data_stack, pinn_net)
print("Plot plotted")


