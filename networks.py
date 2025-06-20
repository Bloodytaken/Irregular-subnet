import torch
import torch.nn as nn

###########################################################
#### Multi layer peceptron/feed forward neural network ####
###########################################################

class MLP_res_net(nn.Module):
    '''Multi-Layer Perceptron with Residual Connection (MLP_res_net) as follows:
              y_pred = net(input) = net_MLP(input) + A * input
              where net_MLP(input) is a simple Multi-Layer Perceptron, e.g.:
                h_1 = input
                h-2 = activation(A_1 h_1 + b_1) #A_1.shape = n_hidden_nodes x input_size
                h_3 = activation(A_2 h_2 + b_2) #A_2.shape = n_hidden_nodes x n_hidden_nodes
                ...
                h_n_hidden_layers = activation(A_n-1 h_n-1 + b_n-1)
                return h_n_hidden_layers
    '''
    def __init__(self, input_size: str | int | list, output_size: str | int | list, n_hidden_layers = 2, n_hidden_nodes = 128, \
                 activation=nn.ReLU, zero_bias=True):
        self.input_size = input_size
        self.output_size = output_size
        super().__init__()
        self.scalar_output = output_size=='scalar'
        #convert input shape:
        def to_num(s):
            if isinstance(s, int):
                return s
            if s=='scalar':
                return 1
            a = 1
            for si in s:
                a = a*(1 if si=='scalar' else si)
            return a
        if isinstance(input_size, list):
            input_size = sum(to_num(s) for s in input_size)
        
        output_size = 1 if self.scalar_output else output_size
        self.net_res = nn.Linear(input_size, output_size)

        seq = [nn.Linear(input_size,n_hidden_nodes),activation()]
        for i in range(n_hidden_layers-1):
            seq.append(nn.Linear(n_hidden_nodes,n_hidden_nodes))
            seq.append(activation())
        seq.append(nn.Linear(n_hidden_nodes,output_size))
        self.net_nonlin = nn.Sequential(*seq)

        if zero_bias:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.constant_(m.bias, val=0) #bias
        
    def forward(self, *ars):
        if len(ars)==1:
            net_in = ars[0]
            net_in = net_in.view(net_in.shape[0], -1) #adds a dim when needed
        else:
            net_in = torch.cat([a.view(a.shape[0], -1) for a in ars],dim=1) #flattens everything
        out = self.net_nonlin(net_in) + self.net_res(net_in)
        return out[:,0] if self.scalar_output else out
    

## MLP res net with time

class MLP_res_net_with_time(nn.Module):
    '''Modified MLP_res_net with time interval (delta_t) as additional input.'''
    def __init__(self, input_size: str | int | list, output_size: str | int | list, n_hidden_layers=2, n_hidden_nodes=64,
                 activation=nn.ReLU, zero_bias=True):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.scalar_output = output_size == 'scalar'

        # Function to convert input shape
        def to_num(s):
            if isinstance(s, int):
                return s
            if s == 'scalar':
                return 1
            a = 1
            for si in s:
                a *= (1 if si == 'scalar' else si)
            return a
        
        if isinstance(input_size, list):
            input_size = sum(to_num(s) for s in input_size)

        output_size = 1 if self.scalar_output else output_size
        self.net_res = nn.Linear(input_size, output_size)

        # Sequential MLP with residual connections
        seq = [nn.Linear(input_size, n_hidden_nodes), activation()]
        for _ in range(n_hidden_layers - 1):
            seq.append(nn.Linear(n_hidden_nodes, n_hidden_nodes))
            seq.append(activation())
        seq.append(nn.Linear(n_hidden_nodes, output_size))
        self.net_nonlin = nn.Sequential(*seq)

        if zero_bias:
            for m in self.modules(): 
                if isinstance(m, nn.Linear):
                    nn.init.constant_(m.bias, val=0)  # Set bias to zero

    def forward(self, *ars):
        if len(ars) == 1:
            net_in = ars[0]
            net_in = net_in.view(net_in.shape[0], -1)  # Adds a dim when needed
        else:
            net_in = torch.cat([a.view(a.shape[0], -1) for a in ars], dim=1)  # Flattens everything

        out = self.net_nonlin(net_in) + self.net_res(net_in)
        return out[:, 0] if self.scalar_output else out
    


# --- Deterministic linear dynamics -----
class LinearAB(torch.nn.Module):
    """dx/dt = A x + B u"""
    def __init__(self, A: torch.Tensor, B: torch.Tensor):
        super().__init__()
        self.register_buffer("A", A)   # freeze 
        self.register_buffer("B", B)

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return x @ self.A.T + u @ self.B.T

class LinearC(torch.nn.Module):
    """ y = C x"""
    def __init__(self, C: torch.Tensor):
        super().__init__()
        self.register_buffer("C", C) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x @ self.C.T
        return y


###########################
###### Integrators ########
###########################

def euler_integrator(f, x, u, dt, n_steps=1):
    dtp = (dt/n_steps)[:,None]
    for _ in range(n_steps): #f(x,u) has shape (nbatch, nx)
        x = x + f(x,u)*dtp
    return x

def rk4_integrator(f, x, u, dt, n_steps=1): # Crucial to modidy n_steps when necessary
    dtp = (dt/n_steps)[:,None]
    for _ in range(n_steps): #f(x,u) has shape (nbatch, nx)
        k1 = dtp * f(x,u)
        k2 = dtp * f(x+k1*0.5,u)
        k3 = dtp * f(x+k2*0.5,u)
        k4 = dtp * f(x+k3,u)
        x = x + (k1+2*k2+2*k3+k4)/6
    return x

def rk45_integrator(f, x, u, dt, n_steps=1):
    dtp = (dt/n_steps)[:,None]
    for _ in range(n_steps): #f(x,u) has shape (nbatch, nx)
        k1 = dtp * f(x, u)
        k2 = dtp * f(x + k1 / 4, u)
        k3 = dtp * f(x + 3 * k1 / 32 + 9 * k2 / 32, u)
        k4 = dtp * f(x + 1932 * k1 / 2197 - 7200 * k2 / 2197 + 7296 * k3 / 2197, u)
        k5 = dtp * f(x + 439 * k1 / 216 - 8 * k2 + 3680 * k3 / 513 - 845 * k4 / 4104, u)
        k6 = dtp * f(x - 8 * k1 / 27 + 2 * k2 - 3544 * k3 / 2565 + 1859 * k4 / 4104 - 11 * k5 / 40, u)
        
        x = x + (16 * k1 / 135 + 6656 * k3 / 12825 + 28561 * k4 / 56430 - 9 * k5 / 50 + 2 * k6 / 55)
    return x

import torch




