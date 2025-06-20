
import nonlinear_benchmarks as nlb
import numpy as np
import torch



class Input_output_data:
    def __init__(self, u, y, sampling_time=None, x=None , name=None, state_initialization_window_length=None,  tau=None): # x=None for test
        assert len(u) == len(y), f'Input u and output y must have the same length, but got {len(u)=}, {len(y)=}'
        if sampling_time is not None:
            assert len(sampling_time) == len(u), f'sampling_time must have the same length as u and y, got {len(sampling_time)=}, {len(u)=}'
        
        self.u = u
        self.y = y
        self.sampling_time = sampling_time
        self.tau = tau
        if self.sampling_time is not None and self.tau is not None:
            self.normalized_time = self.sampling_time / self.tau
        else:
            self.normalized_time = None
            
        self.x = x # test
        self.name = '' if name is None else name
        self.state_initialization_window_length = state_initialization_window_length
    
    def __repr__(self):
        name_str = f' "{self.name}"' if self.name else ''
        st_info = f'sampling_time varying, length={len(self.sampling_time)}' if self.sampling_time is not None else 'sampling_time=None'
        state_win = f', state_initialization_window_length={self.state_initialization_window_length}' if self.state_initialization_window_length else ''
        return f'Input_output_data{name_str} u.shape={self.u.shape}, y.shape={self.y.shape}, {st_info}{state_win}'
    
    def __iter__(self):
        yield self.u
        yield self.y
        yield self.sampling_time
        yield self.x  # for test
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, arg):
        '''Slice the data object in time index'''
        if isinstance(arg, int):
            if arg == 0:
                return self.u
            elif arg == 1:
                return self.y
            elif arg == 2:
                return self.sampling_time
            else:
                raise ValueError(f'Integer index {arg} is invalid. Valid indices: 0 (u), 1 (y), 2 (sampling_time).')
        elif isinstance(arg, slice):
            u_new = self.u[arg]
            y_new = self.y[arg]
            sampling_time_new = self.sampling_time[arg] if self.sampling_time is not None else None
            x_new = self.x[arg] if self.x is not None else None # Test f, h
            # x=x_new for testing f,h.
            return Input_output_data(u_new, y_new, sampling_time_new, x=x_new, name=self.name, state_initialization_window_length=self.state_initialization_window_length)
        else:
            raise ValueError(f'Unsupported argument type "{type(arg)}" for slicing.')
    
    def atleast_2d(self):
        ensure_2d = lambda x: x if x.ndim > 1 else x[:, None]
        sampling_time_copy = self.sampling_time.copy() if self.sampling_time is not None else None
        return Input_output_data(ensure_2d(self.u), ensure_2d(self.y), sampling_time_copy, name=self.name, state_initialization_window_length=self.state_initialization_window_length)
    
    def split(self, frac):
        split_point = int(len(self) * (1 - frac))
        return self[:split_point], self[split_point:]



C = lambda x: torch.as_tensor(x, dtype=torch.float32) if x is not None else None
class IO_normalization_f(torch.nn.Module):
    def __init__(self, fun, umean, ustd):
        super().__init__()
        self.fun, self.umean, self.ustd = fun, C(umean), C(ustd)    
    def forward(self, x, u):
        return self.fun(x, (u-self.umean)/self.ustd)

class IO_normalization_f_CT(torch.nn.Module):
    def __init__(self, fun, umean, ustd, tau):
        super().__init__()
        self.fun, self.umean, self.ustd, self.tau = fun, C(umean), C(ustd), C(tau)
    def forward(self, x, u):
        return self.fun(x, (u-self.umean)/self.ustd)/self.tau

class IO_normalization_h(torch.nn.Module):
    def __init__(self, fun, umean, ustd, ymean, ystd):
        super().__init__()
        self.fun, self.umean, self.ustd, self.ymean, self.ystd = fun, C(umean), C(ustd), C(ymean), C(ystd)
    def forward(self, x, u=None):
        if u is None:
            y_normed = self.fun(x)
        else:
            y_normed = self.fun(x, (u-self.umean)/self.ustd)
        return y_normed*self.ystd + self.ymean

class IO_normalization_encoder(torch.nn.Module):
    def __init__(self, fun, umean, ustd, ymean, ystd):
        super().__init__()
        self.fun, self.umean, self.ustd, self.ymean, self.ystd = fun, C(umean), C(ustd), C(ymean), C(ystd)

    def forward(self, upast, ypast, delta_t_past=None):
        upast_norm = (upast - self.umean) / self.ustd
        ypast_norm = (ypast - self.ymean) / self.ystd
        if delta_t_past is not None:
            return self.fun(upast_norm, ypast_norm, delta_t_past)
        else:
            return self.fun(upast_norm, ypast_norm)


class Norm:
    def __init__(self, umean, ustd, ymean, ystd, sampling_time=1):
        self.umean, self.ustd, self.ymean, self.ystd = C(umean), C(ustd), C(ymean), C(ystd)
        self.sampling_time = C(sampling_time)
    
    def f(self, fun):
        return IO_normalization_f(fun, self.umean, self.ustd)
    def h(self, fun):
        return IO_normalization_h(fun, self.umean, self.ustd, self.ymean, self.ystd)
    def encoder(self, fun):
        return IO_normalization_encoder(fun, self.umean, self.ustd, self.ymean, self.ystd)
    def f_CT(self, fun, tau):
        return IO_normalization_f_CT(fun, self.umean, self.ustd, tau)

    def transform(self, dataset : Input_output_data | list):
        if isinstance(dataset, (list, tuple)):
            return [self.transform(d) for d in dataset]
        u = (dataset.u - self.umean.numpy())/self.ustd.numpy()
        y = (dataset.y - self.ymean.numpy())/self.ystd.numpy()
        sampling_time = None if dataset.sampling_time is None else dataset.sampling_time/self.sampling_time.item()
        return Input_output_data(u, y, sampling_time=sampling_time, name=f'{dataset.name}-normed', \
                                     state_initialization_window_length=dataset.state_initialization_window_length)

    def __repr__(self):
        return (f"Norm(umean={self.umean.numpy()}, ustd={self.ustd.numpy()}, "
                f"ymean={self.ymean.numpy()}, ystd={self.ystd.numpy()}, "
                f"sampling_time={self.sampling_time.numpy()})")

def get_nu_ny_and_auto_norm(data: Input_output_data | list):
    if not isinstance(data, (tuple, list)):
        data = [data]
    u = np.concatenate([d.u for d in data],axis=0)
    y = np.concatenate([d.y for d in data],axis=0)
    assert u.ndim<=2 and y.ndim<=2, f'auto norm only defined for scalar or vector outputs y and input u {y.shape=} {u.shape=}'
    sampling_time = data[0].sampling_time
    #Not useful for irregularly sampling
    # assert all(torch.equal(sampling_time, d.sampling_time) for d in data), \
    #    f"The given datasets don't all have the same sampling_time."
    # assert all(sampling_time==d.sampling_time for d in data), f"the given datasets don't have all the sample sampling_time set {[d.sampling_time for d in data]=}"
    umean, ustd = u.mean(0), u.std(0)
    ymean, ystd = y.mean(0), y.std(0)
    norm = Norm(umean, ustd, ymean, ystd, sampling_time)
    nu = 'scalar' if u.ndim==1 else u.shape[1]
    ny = 'scalar' if y.ndim==1 else y.shape[1]
    return nu, ny, norm





