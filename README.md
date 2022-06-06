# PINN_Thesis

### Save_data.m and evalexpr.m

This code will save the data from the CFD solver to a .mat file. First the x- and y-coordinates are defined which are then used to create a mesh. Using the evalexpr.m function, the data form the CFD solver is extracted for each point on the mesh for different parameters, such as the pressure or the velocity in x- and y-direction. 

### PINN_model.py

This code contains the model for the Physics-Informed Neural Network, which is adapted from a code provided by Shengfeng233. The equations from the original code are changed to fit the steady-state Navier-Stokes equations, which are not dependent on time. This requires the code to further be adapated to fit a time independent data-set. The time-parameters are taken out, as well as any derivatives dependent on time.

This code uses a Python library called PyTorch, specifically made for deep learning. The modules this library contains are important for the creation and training of a neural network. The NN module is used to create the Neural Network. In this model the nn.linear() module is used to apply linear transformation to the input, it takes the size of the current layer and the size of the next layer as input. The activation function used is the Tanh function. The nn.parameters() module is used to obtain the weights and bias.

This model also contains several functions, for example the f_equation_identification function. This function uses the feed forward algorithm in the pinn_net class to predict an outcome for the parameters. Using PyTorch's automatic differentiation module, Autograd, the derivatives for each parameter is obtained. This is then used to construct the Steady-State Navier-Stokes equations. The output of this function is both the predicted outcomes for the parameters u and v, and the equations.

### train.py

This part is used to train the model, which aims to reduce the loss of both the predicted values and the equations. Using the f_equation_identification function, the predicted outcomes and the equations are determined. The loss function is determined using the nn.MSEloss module, which then needs to be optimized. This is done using the Adam optimizer, a PyTorch Package. 

The losses of both the predicted values and the equations, as well as the total loss, are all saved to a CSV file. 

### plot.py

Both the input data and the predicted outcomes are plotted in order to accurately portray the difference for both simulations. The x- and y-coordinates are defined using the same method as in the Save_data.m file, in order to accurately generate the mesh. The parameters are obtained by calling the f_equation_inverse function in PINN_model.py. This will give as output the predicted data for the parameters, using the same method as the f_equation_identification. 

This output is then reshaped to fit the x by y mesh, and is then plotted using the plot_compare function. This plots the velocity in both x- and y-direction as well as the pressure.

