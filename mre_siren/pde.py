import torch


def as_matrix(x):
    '''
    Args:
        x: A tensor with any number of dims.
    Returns:
        x: The same tensor reshaped as a matrix,
            assuming the first dim is a batch dim.
    '''
    if x.ndim == 0: # scalar
        return x.broadcast_to(1, 1)
    elif x.ndim == 1: # vector
        return x.unsqueeze(1)
    elif x.ndim > 2: # higher rank tensor
        return x.reshape(x.shape[0], -1)
    else: # already a matrix
        return x


def jacobian(y, x):
    '''
    Args:
        y: (N, M) tensor of outputs.
        x: (N, K) tensor of inputs.
    Returns:
        jac: (N, M, K) Jacobian tensor
            where jac_ijk = dy_ij/dx_ik.
    '''
    x, y = as_matrix(x), as_matrix(y)
    N, M = y.shape
    N, K = x.shape
    assert y.shape == (N, M), y.shape
    ones = torch.ones_like(y[:,0])
    jac = []
    for j in range(M):
        # each row of the Jacobian is the gradient of a y component wrt x
        grad = torch.autograd.grad(y[:,j], x, grad_outputs=ones, create_graph=True, allow_unused=True)[0]
        if grad is None:
            grad = torch.zeros_like(x)
        jac.append(grad)
    
    jac = torch.stack(jac, dim=1)
    assert jac.shape == (N, M, K), jac.shape
    return jac


def divergence(y, x):
    '''
    Args:
        y: (N, K) tensor of outputs.
        x: (N, K) tensor of inputs.
    Returns:
        div: (N,) divergence tensor
            where div_i = sum_j dy_ij/dx_ij.
    '''
    x, y = as_matrix(x), as_matrix(y)
    N, K = x.shape
    assert y.shape == (N, K), y.shape
    jac = jacobian(y, x)
    assert jac.shape == (N, K, K), jac.shape
    div = 0
    for j in range(K):
        # divergence is the trace of the Jacobian
        div = div + jac[:,j,j]
    return div


def laplacian(y, x):
    '''
    Args:
        y: (N, M) tensor of outputs.
        x: (N, K) tensor of inputs.
    Returns:
        lap: (N, M) Laplacian tensor,
            where lap_ij = sum_k d^2 y_ij/dx_ik^2.
    '''
    shape = y.shape
    x, y = as_matrix(x), as_matrix(y)
    N, M = y.shape
    N, K = x.shape
    assert y.shape == (N, M), y.shape
    jac = jacobian(y, x)
    assert jac.shape == (N, M, K), jac.shape
    lap = []
    for j in range(M):
        # the Laplacian is the divergence of the gradient
        #   i.e. lap_ij = sum_k d^2 y_ij / dx_ik^2
        lap.append(divergence(jac[:,j,:], x))
    return torch.stack(lap, dim=1).reshape(*shape)


def compute_pde_loss(x, u, G, frequency, density):
    
    n = x.shape[0]
    u = u.reshape(n, -1, 2)
    u_real = u[...,0]
    u_imag = u[...,1]
    abs_u =  torch.sqrt(u_real**2 + u_imag**2)
    
    laplace_u_real = laplacian(u_real, x)
    laplace_u_imag = laplacian(u_imag, x)
    abs_laplace_u = torch.sqrt(laplace_u_real**2 + laplace_u_imag**2)
    #print(abs_u.shape, u_real.shape, u_imag.shape)

    abs_G = torch.abs(G[:,0:1])
    cos_G = torch.cos(G[:,1:2])
    #print(abs_G.shape, cos_G.shape)
    
    abs_G_error = abs_laplace_u*abs_G - density*(2*np.pi*frequency)**2*abs_u 
    cos_G_error = abs_laplace_u*abs_u*cos_G + laplace_u_real*u_real + laplace_u_imag*u_imag
    #print(abs_G_error.shape, cos_G_error.shape)

    return (abs_G_error.abs()**2 + cos_G_error.abs()**2).mean()
