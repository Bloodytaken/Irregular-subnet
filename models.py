from normalization import Input_output_data
import numpy as np
import torch
from normalization import Norm
from networks import MLP_res_net_with_time
from networks import MLP_res_net
from torch import nn

###################
##Helper function##
###################


import numpy as np
import torch
from typing import Tuple, Union, List

def past_future_arrays(data: Input_output_data | list, na: int, nb: int, T: int | str, stride: int = 1, add_sampling_time: bool = False):
    """
    return:
      - (upast, ypast, ufuture, yfuture, [delta_t_past, delta_t_future]) if not add_sampling_time,
      - (upast, ypast, ufuture, sampling_time, yfuture, delta_t_past, delta_t_future) if add_sampling_time
      - ids (np.ndarray)
    """
    if T == 'sim':
        if isinstance(data, (tuple, list)):
            assert all(len(data[0]) == len(d) for d in data), "if T='sim' then all datasets must have the same length"
            T = len(data[0]) - max(na, nb)
        else:
            T = len(data) - max(na, nb)

    if isinstance(data, (tuple, list)):
        u = np.concatenate([di.u.numpy() if isinstance(di.u, torch.Tensor) else di.u for di in data]).astype(np.float32)
        y = np.concatenate([di.y.numpy() if isinstance(di.y, torch.Tensor) else di.y for di in data]).astype(np.float32)
        delta_t = np.concatenate([
            (di.delta_t if hasattr(di, 'delta_t') else di.sampling_time).numpy()
            if isinstance(di.delta_t if hasattr(di, 'delta_t') else di.sampling_time, torch.Tensor)
            else (di.delta_t if hasattr(di, 'delta_t') else di.sampling_time)
            for di in data
        ]).astype(np.float32)
    else:
        u = data.u.numpy().astype(np.float32) if isinstance(data.u, torch.Tensor) else data.u.astype(np.float32)
        y = data.y.numpy().astype(np.float32) if isinstance(data.y, torch.Tensor) else data.y.astype(np.float32)
        delta_t_attr = data.delta_t if hasattr(data, 'delta_t') else data.sampling_time
        delta_t = (
            delta_t_attr.numpy().astype(np.float32)
            if isinstance(delta_t_attr, torch.Tensor)
            else delta_t_attr.astype(np.float32)
        ).ravel() 
        
        # delta_t = delta_t_attr.numpy().astype(np.float32) if isinstance(delta_t_attr, torch.Tensor) else delta_t_attr.astype(np.float32)

    def window(x, window_shape):
        x = np.lib.stride_tricks.sliding_window_view(x, window_shape=window_shape, axis=0)
        s_window = (0, len(x.shape) - 1) + tuple(range(1, len(x.shape) - 1))
        return x.transpose(s_window)

    npast = max(na, nb)
    ufuture = window(u[npast:len(u)], window_shape=T)
    yfuture = window(y[npast:len(y)], window_shape=T)

    delta_t_shifted = np.concatenate(
        [delta_t[1:], np.array([0.], dtype=np.float32)]
    ).astype(np.float32)
    upast = window(u[npast - nb : len(u) - T], window_shape=nb)[:, ::-1].copy()
    ypast = window(y[npast - na : len(y) - T], window_shape=na)[:, ::-1].copy()

    delta_t_past = window(
        delta_t_shifted[npast - npast : len(delta_t_shifted) - T], 
        window_shape=npast
    )[:, ::-1].copy()
    # delta_t_past[delta_t_past == 0] = 1e-9
    delta_t_future = window(delta_t_shifted[npast: len(delta_t_shifted)], window_shape=T)


    if isinstance(data, (tuple, list)):
        acc_L, ids = 0, []
        for d in data:
            assert len(d.u) >= npast + T, f"not enough length: {len(d.u)} < {npast + T}"
            ids.append(np.arange(0, len(d.u) - npast - T + 1, stride) + acc_L)
            acc_L += len(d.u)
        ids = np.concatenate(ids)
    else:
        ids = np.arange(0, len(data) - npast - T + 1, stride)

    s = lambda x: torch.tensor(x, dtype=torch.float32)
    if not add_sampling_time:
        return (s(upast), s(ypast), s(ufuture), s(yfuture), s(delta_t_past), s(delta_t_future)), ids
    else:
        sampling_time = window(delta_t[npast:len(delta_t)], window_shape=T)
        return (s(upast), s(ypast), s(ufuture), s(sampling_time), s(yfuture), s(delta_t_past), s(delta_t_future)), ids


def validate_SUBNET_ISTS_structure(model):
    nx, nu, ny, na, nb = model.nx, model.nu, model.ny, model.na, model.nb
    v = lambda *size: torch.randn(size)
    xtest = v(1, nx)
    utest = v(1) if nu == 'scalar' else v(1, nu)
    upast_test = v(1, nb) if nu == 'scalar' else v(1, nb, nu)
    ypast_test = v(1, na) if ny == 'scalar' else v(1, na, ny)
    delta_t_past_test = v(1, max(na, nb), 1)  # Time-related input for the encoder

    with torch.no_grad():
        if isinstance(model, SUBNET_ISTS):
            # Testing the f function
            f = model.f
            xnext_test = f(xtest, utest)
            assert xnext_test.shape == (1, nx), f'f function returned incorrect shape, expected (1, nx) but got {xnext_test.shape}'

            # Testing the encoder with time inputs
            x_encoded = model.encoder(upast_test, ypast_test, delta_t_past_test)
            assert x_encoded.shape == (1, nx), f'Encoder returned incorrect shape, expected (1, nx) but got {x_encoded.shape}'

            # Testing the h function
            y_pred = model.h(xtest, utest) if model.feedthrough else model.h(xtest)
            assert (y_pred.shape == (1,)) if ny == 'scalar' else (y_pred.shape == (1, ny)), f'h function returned incorrect shape, expected shape was not met'

            # Testing the integrator if present
            if model.integrator:
                dt_test = torch.ones((1,))
                xnext_test = model.integrator(model.f, xtest, utest, dt_test)
                assert xnext_test.shape == (1, nx), f'Integrator returned incorrect shape, expected (1, nx) but got {xnext_test.shape}'
        else:
            raise NotImplementedError(f'Model validation for type {type(model)} cannot be validated yet')


####################
####SUBNET_ISTS#####
####################
    
class SUBNET_ISTS(nn.Module):
    def __init__(self, nu:int|str, ny:int|str, norm : Norm, nx:int=10, nb:int=20, na:int=20, \
                 f=None, h=None, encoder=None, integrator = None, feedthrough=False, validate=True) -> None:
        super().__init__()
        self.nu, self.ny, self.norm, self.nx, self.nb, self.na, self.feedthrough = nu, ny, norm, nx, nb, na, feedthrough
        self.integrator = integrator

        self.f = f if f is not None else norm.f(MLP_res_net(input_size = [nx , nu], output_size = nx))
        self.h = h if h is not None else norm.h(MLP_res_net(input_size = [nx , nu] if feedthrough else nx, output_size = ny))

        # Use 'MLP_res_net_with_time' in Encoder
        encoder_input_size = [(nb, nu), (na, ny), (max(na, nb), 1)]  # delta_t_past: shape=(max(na,nb), 1)
        self.encoder = encoder if encoder is not None else \
            norm.encoder(MLP_res_net_with_time(input_size=encoder_input_size, output_size=nx))
        
        
        if validate:
            validate_SUBNET_ISTS_structure(self)
    
    def create_arrays(self, data, T=50, stride=1, *, return_x0=False):
       
        (up, yp, uf, yf, dtp, dtf), ids = past_future_arrays(
            data, self.na, self.nb, T=T, stride=stride, add_sampling_time=False)
        if not return_x0:       # For testing below     
            return (up, yp, uf, yf, dtp, dtf), ids
        T_eff  = uf.shape[1]
        npast  = max(self.na, self.nb)

        def get_x(d):

            return d.x.numpy() if isinstance(d.x, torch.Tensor) else d.x

        x_full = (np.concatenate([get_x(d) for d in data], axis=0)
                if isinstance(data, (tuple, list)) else get_x(data))

        x0_all  = x_full[npast-1 : len(x_full)-T_eff]
        x0_true = torch.tensor(x0_all[ids], dtype=torch.float32)

        return (up, yp, uf, yf, dtp, dtf, x0_true), ids


    def forward(self, 
                upast: torch.Tensor, ypast: torch.Tensor, ufuture: torch.Tensor, yfuture: torch.Tensor,  delta_t_past: torch.Tensor, 
                delta_t_future: torch.Tensor, sampling_time: torch.Tensor = None, x0_override: torch.Tensor | None = None,# test h,f
                use_encoder: bool = True ): # test f,h
   
        # 1. Encoder：upast, ypast, delta_t_past => initial state x
        # x = self.encoder(upast, ypast, delta_t_past)
        if use_encoder:
            if x0_override is None:
                x = self.encoder(upast, ypast, delta_t_past)      # (B , nx)
            else:
                x = x0_override
        else:  
            if x0_override is None:
                raise ValueError("use_encoder=False")
            x = x0_override

        xfuture = []
        # 2.  ufuture + delta_t_future => integrator
        for t, u in enumerate(ufuture.swapaxes(0, 1)):
            xfuture.append(x)
            dt = delta_t_future[:, t]
            if dt.dim() > 1:
                dt = dt.squeeze(-1)
            x = self.integrator(self.f, x, u, dt)
        # (B, T, nx)
        xfuture = torch.stack(xfuture, dim=1)    
        # 3. future outputs
        fl = lambda ar: torch.flatten(ar, start_dim=0, end_dim=1)  # (B, T, *) -> (B*T, *)
        if self.feedthrough:
            yfuture_sim_flat = self.h(fl(xfuture), fl(ufuture))#(B*T, nu)
        else:
            yfuture_sim_flat = self.h(fl(xfuture))
        # recover (B, T, ny) 
        yfuture_sim = torch.unflatten(yfuture_sim_flat, dim=0, sizes=(ufuture.shape[0], ufuture.shape[1]))
        
        return yfuture_sim

    def simulate(self, data: Input_output_data | list):
        if isinstance(data, (list, tuple)):
            return [self.simulate(d) for d in data]
        ysim = self(*past_future_arrays(data, self.na, self.nb, T='sim', add_sampling_time=False)[0])[0].detach().numpy()
        return Input_output_data(
            u=data.u, 
            y=np.concatenate([data.y[:max(self.na, self.nb)], ysim], axis=0), 
            sampling_time=data.sampling_time,
            state_initialization_window_length=max(self.na, self.nb)
        )

    def f_unbached(self, x, u):
        return self.f(x[None], u[None])[0]

    def h_unbached(self, x, u=None):
        return self.h(x[None], u[None])[0] if self.feedthrough else self.h(x[None])[0]

    def encoder_unbached(self, upast, ypast, delta_t_past):
        return self.encoder(upast[None], ypast[None], delta_t_past[None])[0]
