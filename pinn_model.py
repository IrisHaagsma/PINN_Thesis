import scipy.io
import numpy as np
import torch
import torch.nn as nn
import pandas as pd

# Define PINN network module, including data read function, parameter initialization
# Bias and derivation functions for forward and inverse problems
# global parameters
filename_load_model = './NS_model_train_0.02_3.0.pt'
filename_save_model = './NS_model_train_0.02_3.0.pt'
filename_data = './CFD_0.02_3.0.mat'
filename_loss = './loss_0.02_3.0.csv'
# Whether the training device is a GPU or a CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Read the original data and convert it to x, y,--u, v, p, the return value is Tensor type
def read_data(filename):
    # read raw data
    data_mat = scipy.io.loadmat(filename)
    U_star = data_mat['U_star']
    X_star = data_mat['X_star']
    P_star = data_mat['P_star']



    # Read the number of coordinate points N and the number of time steps T
    N = X_star.shape[0]
    X = np.linspace(0.25, 2.2, 100)
    Y = np.linspace(0.0, 0.41, 100)
    [ys, xs] = np.meshgrid(Y, X)

    x = np.ndarray.flatten(xs).reshape(-1, 1)
    y = np.ndarray.flatten(ys).reshape(-1, 1)

    UU = U_star[:, 0]
    VV = U_star[:, 1]
    PP = P_star

    u = UU.flatten()[:, None]
    v = VV.flatten()[:, None]
    p = PP.flatten()[:, None]
    temp = np.concatenate((x, y, u, v, p), 1)
    feature_mat = np.empty((2, 5))
    feature_mat[0, :] = np.nanmax(temp, 0)
    feature_mat[1, :] = np.nanmin(temp, 0)
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    u = torch.tensor(u, dtype=torch.float32)
    v = torch.tensor(v, dtype=torch.float32)
    p = torch.tensor(p, dtype=torch.float32)
    feature_mat = torch.tensor(feature_mat, dtype=torch.float32)
    return x, y, u, v, p, N, feature_mat


def read_data_portion(filename, portion):
    # read raw data
    data_mat = scipy.io.loadmat(filename)
    U_star = data_mat['U_star']
    X_star = data_mat['X_star']
    P_star = data_mat['P_star']

    # Read the number of coordinate points N and the number of time steps T
    N = X_star.shape[0]

    X = np.linspace(0.25, 2.2, 100)
    Y = np.linspace(0.0, 0.41, 100)
    [ys,xs] = np.meshgrid(Y,X)

    UU = U_star[:, 0]
    VV = U_star[:, 1]
    PP = P_star


    x= np.ndarray.flatten(xs).reshape(-1, 1)
    y= np.ndarray.flatten(ys).reshape(-1, 1)
    u = UU.flatten()[:, None]
    v = VV.flatten()[:, None]
    p = PP.flatten()[:, None]
    temp = np.concatenate((x,y,u,v,p),1)

    feature_mat = np.empty((2, 5))
    feature_mat[0, :] = np.nanmax(temp, 0)
    feature_mat[1, :] = np.nanmin(temp, 0)
    x_unique = np.unique(x).reshape(-1, 1)
    y_unique = np.unique(y).reshape(-1, 1)
    index_arr_x = np.linspace(0, len(x_unique)-1, int(len(x_unique) * portion)).astype(int).reshape(-1, 1)
    index_arr_y = np.linspace(0, len(y_unique)-1, int(len(y_unique) * portion)).astype(int).reshape(-1, 1)
    x_select = x_unique[index_arr_x].reshape(-1, 1)
    y_select = y_unique[index_arr_y].reshape(-1, 1)
    del x_unique, y_unique, index_arr_x, index_arr_y
    index_x = np.empty((0, 1), dtype=int)
    index_y = np.empty((0, 1), dtype=int)
    for select_1 in x_select:
        index_x = np.append(index_x, np.where(x == select_1)[0].reshape(-1, 1), 0)
    for select_2 in y_select:
        index_y = np.append(index_y, np.where(y == select_2)[0].reshape(-1, 1), 0)
    index_all = np.intersect1d(index_x, index_y, assume_unique=False, return_indices=False).reshape(-1, 1)
    x = x[index_all].reshape(-1, 1)
    y = y[index_all].reshape(-1, 1)

    u = u[index_all].reshape(-1, 1)
    v = v[index_all].reshape(-1, 1)
    p = p[index_all].reshape(-1, 1)
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    u = torch.tensor(u, dtype=torch.float32)
    v = torch.tensor(v, dtype=torch.float32)
    p = torch.tensor(p, dtype=torch.float32)
    feature_mat = torch.tensor(feature_mat, dtype=torch.float32)
    return x, y, u, v, p, feature_mat


def read_data_portion_export(filename, portion):
    # read raw data
    data_mat = scipy.io.loadmat(filename)
    U_star = data_mat['U_star']
    X_star = data_mat['X_star']
    P_star = data_mat['P_star']

    # Read the number of coordinate points N
    N = X_star.shape[0]


    # Convert the data to x,y,t---u,v,p(N*T,1)
    XX = X_star[:, 0]
    YY = X_star[:, 1]
    UU = U_star[:, 0]
    VV = U_star[:, 1]
    PP = P_star
    x = XX.flatten()[:, None]
    y = YY.flatten()[:, None]
    u = UU.flatten()[:, None]
    v = VV.flatten()[:, None]
    p = PP.flatten()[:, None]
    temp = np.concatenate((x,y,u,v,p),1)
    feature_mat = np.empty((2, 5))
    feature_mat[0, :] = np.nanmax(temp, 0)
    feature_mat[1, :] = np.nanmin(temp, 0)
    x_unique = np.sort(np.unique(x).reshape(-1, 1))
    y_unique = np.sort(np.unique(y).reshape(-1, 1))
    portion_x = int(len(x_unique)*portion)
    portion_y = int(len(y_unique)*portion)
    x_select = x_unique[0:portion_x, 0].reshape(-1, 1)
    y_select = y_unique[0:portion_y, 0].reshape(-1, 1)
    del x_unique, y_unique
    index_x = np.empty((0, 1), dtype=int)
    index_y = np.empty((0, 1), dtype=int)
    for select_1 in x_select:
        index_x = np.append(index_x, np.where(x == select_1)[0].reshape(-1, 1), 0)
    for select_2 in y_select:
        index_y = np.append(index_y, np.where(y == select_2)[0].reshape(-1, 1), 0)
    index_all = np.intersect1d(index_x, index_y, assume_unique=False, return_indices=False).reshape(-1, 1)
    x = x[index_all].reshape(-1, 1)
    y = y[index_all].reshape(-1, 1)
    u = u[index_all].reshape(-1, 1)
    v = v[index_all].reshape(-1, 1)
    p = p[index_all].reshape(-1, 1)
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    u = torch.tensor(u, dtype=torch.float32)
    v = torch.tensor(v, dtype=torch.float32)
    p = torch.tensor(p, dtype=torch.float32)
    feature_mat = torch.tensor(feature_mat, dtype=torch.float32)
    return x, y, u, v, p, feature_mat


# def read_data_part_time(filename, portion):
    # read raw data
    # data_mat = scipy.io.loadmat(filename)
    # U_star = data_mat['U_star']  # N*dimension*T
    # X_star = data_mat['X_star']  # N*dimension
    # T_star = data_mat['t']  # T*1
    # P_star = data_mat['p_star']  # N*T
    #
    # # Read the number of coordinate points N and the number of time steps T
    # N = X_star.shape[0]
    # T = T_star.shape[0]
    #
    # # Convert the data to x,y,t---u,v,p(N*T,1)
    # XX = np.tile(X_star[:, 0:1], (1, T))
    # YY = np.tile(X_star[:, 1:2], (1, T))
    # TT = np.tile(T_star, (1, N)).T
    # UU = U_star[:, 0, :]
    # VV = U_star[:, 1, :]
    # PP = P_star
    # x = XX.flatten()[:, None]
    # y = YY.flatten()[:, None]
    # t = TT.flatten()[:, None]
    # u = UU.flatten()[:, None]
    # v = VV.flatten()[:, None]
    # p = PP.flatten()[:, None]
    # temp = np.concatenate((x,y,t,u,v,p),1)
    # feature_mat = np.empty((2, 6))
    # feature_mat[0, :] = np.max(temp, 0)
    # feature_mat[1, :] = np.min(temp, 0)
    # t_unique = np.unique(t).reshape(-1, 1)
    # index_arr_t = np.linspace(0, len(t_unique)-1, int(len(t_unique) * portion)).astype(int).reshape(-1, 1)
    # t_select = t_unique[index_arr_t].reshape(-1, 1)
    # del t_unique, index_arr_t
    # index_t = np.empty((0, 1), dtype=int)
    # for select_1 in t_select:
    #     index_t = np.append(index_t, np.where(t == select_1)[0].reshape(-1, 1), 0)
    # x = x[index_t].reshape(-1, 1)
    # y = y[index_t].reshape(-1, 1)
    # t = t[index_t].reshape(-1, 1)
    # u = u[index_t].reshape(-1, 1)
    # v = v[index_t].reshape(-1, 1)
    # p = p[index_t].reshape(-1, 1)
    # x = torch.tensor(x, dtype=torch.float32)
    # y = torch.tensor(y, dtype=torch.float32)
    # t = torch.tensor(t, dtype=torch.float32)
    # u = torch.tensor(u, dtype=torch.float32)
    # v = torch.tensor(v, dtype=torch.float32)
    # p = torch.tensor(p, dtype=torch.float32)
    # feature_mat = torch.tensor(feature_mat, dtype=torch.float32)
    # return x, y, t, u, v, p, feature_mat


# Define the network structure, specify the number of network layers and neurons by the layer list
class PINN_Net(nn.Module):
    def __init__(self, layer_mat):
        super(PINN_Net, self).__init__()
        self.layer_num = len(layer_mat) - 1
        self.base = nn.Sequential()
        for i in range(0, self.layer_num - 1):
            self.base.add_module(str(i) + "linear", nn.Linear(layer_mat[i], layer_mat[i + 1]))
            # nn.init.kaiming_normal()
            self.base.add_module(str(i) + "Act", nn.Tanh())
        self.base.add_module(str(self.layer_num - 1) + "linear",
                             nn.Linear(layer_mat[self.layer_num - 1], layer_mat[self.layer_num]))
        self.lam1 = nn.Parameter(torch.randn(1, requires_grad=True))
        self.lam2 = nn.Parameter(torch.randn(1, requires_grad=True))
        self.Initial_param()

    def forward(self, x, y):
        X = torch.cat([x, y], 1).requires_grad_(True)
        predict = self.base(X)
        return predict

    # Initialize parameters
    def Initial_param(self):
        for name, param in self.base.named_parameters():
            if name.endswith('weight'):
                nn.init.xavier_normal_(param)
            elif name.endswith('bias'):
                nn.init.zeros_(param)


# Define the partial differential equation (the deviation of the) inverse as the inverse problem
def f_equation_inverse(x, y, pinn_example):
    lam1 = pinn_example.lam1
    lam2 = pinn_example.lam2
    predict_out = pinn_example.forward(x, y)
    # get the predicted output psi,p
    psi = predict_out[:, 0].reshape(-1, 1)
    p = predict_out[:, 1].reshape(-1, 1)
    # Calculate each partial derivative through automatic differentiation, 
    # where .sum() converts the vector into a scalar, which has no practical significance
    u = torch.autograd.grad(psi.sum(), y, create_graph=True)[0]
    v = -torch.autograd.grad(psi.sum(), x, create_graph=True)[0]
    # u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
    u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    u_y = torch.autograd.grad(u.sum(), y, create_graph=True)[0]
    # v_t = torch.autograd.grad(v.sum(), t, create_graph=True)[0]
    v_x = torch.autograd.grad(v.sum(), x, create_graph=True)[0]
    v_y = torch.autograd.grad(v.sum(), y, create_graph=True)[0]
    p_x = torch.autograd.grad(p.sum(), x, create_graph=True)[0]
    p_y = torch.autograd.grad(p.sum(), y, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y.sum(), y, create_graph=True)[0]
    v_xx = torch.autograd.grad(v_x.sum(), x, create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y.sum(), y, create_graph=True)[0]
    # Compute residuals of partial differential equations
    f_equation_x = (u * u_x + v * v_x) + 1/lam1 * p_x - lam2 * (u_xx + u_yy)
    f_equation_y = (u * v_x + v * v_y) + 1/lam1 * p_y - lam2 * (v_xx + v_yy)
    return u, v, p, f_equation_x, f_equation_y


def f_equation_identification(x, y, pinn_example, lam1=1.0, lam2=0.01):
    # Positive problem, the user needs to provide the parameter value of the system, 
    # the default is 1&0.01
    predict_out = pinn_example.forward(x, y)
    # get the predicted output psi,p
    psi = predict_out[:, 0].reshape(-1, 1)
    p = predict_out[:, 1].reshape(-1, 1)
    # Calculate each partial derivative through automatic differentiation,
    # where .sum() converts the vector into a scalar, which has no practical significance
    u = torch.autograd.grad(psi.sum(), y, create_graph=True)[0]
    v = -torch.autograd.grad(psi.sum(), x, create_graph=True)[0]
    # u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
    u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    u_y = torch.autograd.grad(u.sum(), y, create_graph=True)[0]
    # v_t = torch.autograd.grad(v.sum(), t, create_graph=True)[0]
    v_x = torch.autograd.grad(v.sum(), x, create_graph=True)[0]
    v_y = torch.autograd.grad(v.sum(), y, create_graph=True)[0]
    p_x = torch.autograd.grad(p.sum(), x, create_graph=True)[0]
    p_y = torch.autograd.grad(p.sum(), y, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y.sum(), y, create_graph=True)[0]
    v_xx = torch.autograd.grad(v_x.sum(), x, create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y.sum(), y, create_graph=True)[0]
    # Compute residuals of partial differential equations
    f_equation_x = (u * u_x + v * u_y) + 1/lam1 * p_x - lam2 * (u_xx + u_yy)
    f_equation_y = (u * v_x + v * v_y) + 1/lam1 * p_y - lam2 * (v_xx + v_yy)
    return u, v, f_equation_x, f_equation_y


def f_equation_inverse_simple_norm(x, y, feature_mat, pinn_example):
    lam1 = pinn_example.lam1
    lam2 = pinn_example.lam2
    predict_out = pinn_example.forward(x, y)
    # get the predicted output psi,p
    u = predict_out[:, 0].reshape(-1, 1)
    v = predict_out[:, 1].reshape(-1, 1)
    p = predict_out[:, 2].reshape(-1, 1)
    # Calculate each partial derivative through automatic differentiation, 
    # where .sum() converts the vector into a scalar, which has no practical significance
    # u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
    u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    u_y = torch.autograd.grad(u.sum(), y, create_graph=True)[0]
    # v_t = torch.autograd.grad(v.sum(), t, create_graph=True)[0]
    v_x = torch.autograd.grad(v.sum(), x, create_graph=True)[0]
    v_y = torch.autograd.grad(v.sum(), y, create_graph=True)[0]
    p_x = torch.autograd.grad(p.sum(), x, create_graph=True)[0]
    p_y = torch.autograd.grad(p.sum(), y, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y.sum(), y, create_graph=True)[0]
    v_xx = torch.autograd.grad(v_x.sum(), x, create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y.sum(), y, create_graph=True)[0]
    # Since the data is normalized, the equation needs to be modified
    # Calculate the residuals of partial differential equations,
    # including three equations including continuity equation and momentum equation
    [L1, L2, U1, U2, p0] = feature_mat[0, :]
    f_equation_c = u_x * U1 / L1 + v_y * U2 / L2
    f_equation_x = u * u_x * U1 * U1 / L1 + v * u_y * U1 * U2 / L2 + lam1 * p_x * p0 / L1 - lam2 * (
                u_xx * U1 / (L1 * L1) + u_yy * U1 / (L2 * L2))
    f_equation_y = u * v_x * U1 * U2 / L1 + v * v_y * U2 * U2 / L2 + lam1 * p_y * p0 / L2 - lam2 * (
                v_xx * U2 / (L1 * L1) + v_yy * U2 / (L2 * L2))
    # [L1, L2, t0, U1, U2, p0] = feature_mat[0, :]
    # f_equation_c = u_x*U1/L1+v_y*U2/L2
    # f_equation_x = u * u_x*U1*U1/L1 + v * u_y*U1*U2/L2 + lam1 - lam2 * (u_xx*U1/(L1*L1) + u_yy*U1/(L2*L2))
    # f_equation_y = u * v_x*U1*U2/L1 + v * v_y*U2*U2/L2 + lam1 - lam2 * (v_xx*U2/(L1*L1) + v_yy*U2/(L2*L2))
    return u, v, f_equation_c, f_equation_x, f_equation_y


def shuffle_data(x, y, u, v, p):
    X_total = torch.cat([x, y, u, v, p], 1)
    X_total_arr = X_total.data.numpy()
    X_total_arr = np.nan_to_num(X_total_arr)
    #X_total_arr = np.ma.masked_invalid(X_total_arr)
    np.random.shuffle(X_total_arr)
    X_total_random = torch.tensor(X_total_arr)
    return X_total_random


def simple_norm(x, y, u, v, p, feature_mat):
    x = x / feature_mat[0, 0]
    y = y / feature_mat[0, 1]
    # t = t / feature_mat[0, 2]
    u = u / feature_mat[0, 3]
    v = v / feature_mat[0, 4]
    p = p / feature_mat[0, 5]
    return x, y, u, v, p, feature_mat
