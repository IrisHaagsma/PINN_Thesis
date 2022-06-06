% Create evenly spaced x- and y- coordinates
X = linspace(0.25,2.2,100);
Y = linspace(0.0,0.41,100);

% generate mesh
[XX,YY] = meshgrid(X,Y);


M = [XX(:),YY(:)]';



% Cell array of expressions to evaluate.
x_data = { 'x', 'y' };

% Evaluate expressions and store results in data array.
X_star = zeros( size(M,2), length(x_data) );
for i=1:length(x_data)
  X_star(:,i) = evalexpr( x_data{i}, M, fea );
end
% 
% Cell array of expressions to evaluate.
U_data = { 'u', 'v' };


% Evaluate expressions and store results in data array.
U_star = zeros( size(M,2), length(U_data) );
for i=1:length(U_data)
  U_star(:,i) = evalexpr( U_data{i}, M, fea );
end

% Cell array of expressions to evaluate
press_data = { 'p' };


% Evaluate expressions and store results in data array.
P_star = zeros( size(M,2), length(press_data) );
for i=1:length(press_data)
  P_star(:,i) = evalexpr( press_data{i}, M, fea );
end

% Cell array of expressions to evaluate.
vel_data = { 'sqrt(u^2+v^2)' };


% Evaluate expressions and store results in data array.
velocity = zeros( size(M,2), length(vel_data) );
for i=1:length(vel_data)
  velocity(:,i) = evalexpr( vel_data{i}, M, fea );
end
% 
% Example to save the data for simulation with a grid size of 0.01 and a maximum velocity of 15.0
save('CFD_0.01_15.0.mat', 'X_star', 'U_star', 'P_star', 'velocity' )
