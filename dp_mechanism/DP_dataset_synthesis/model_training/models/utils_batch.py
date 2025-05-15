import torch

def batch_to_gpu(params, batch_size=100):
    gpu_params = {}
    items = list(params.items())
    for i in range(0, len(items), batch_size):
        batch = items[i:i+batch_size]
        for k, v in batch:
            gpu_params[k] = v.cuda() if torch.cuda.is_available() else v
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    return gpu_params

def cast(params, dtype):
    if isinstance(params, dict):
        return {k: cast(v, dtype) for k,v in params.items()}
    return getattr(params.cuda() if torch.cuda.is_available() else params, dtype)()

def flatten(params):
    flat_params = {}
    for k, v in params.items():
        if isinstance(v, dict):
            for inner_k, inner_v in flatten(v).items():
                flat_params[f"{k}.{inner_k}"] = inner_v
        else:
            flat_params[k] = v
    return batch_to_gpu(flat_params)

def conv_params(k, s, c):
    shape = {'weight': (s, c, k, k), 'bias': (s,)}
    params = {name: torch.zeros(shape, dtype=torch.float32) for name, shape in shape.items()}
    return params

def bnparams(n):
    shape = {'weight': (n,), 'bias': (n,), 'running_mean': (n,), 'running_var': (n,)}
    params = {name: torch.zeros(shape, dtype=torch.float32) for name, shape in shape.items()}
    params['weight'].fill_(1)
    params['bias'].zero_()
    params['running_mean'].zero_()
    params['running_var'].fill_(1)
    return params