import torch


def jacobian(y, x):
    '''
    Args:
        y: (N, M) tensor of outputs.
        x: (N, K) tensor of inputs.
    Returns:
        jac: (N, M, K) Jacobian tensor
            where jac_ijk = dy_ij/dx_ik.
    '''
    N, M = y.shape
    N, K = x.shape
    assert y.shape == (N, M), y.shape
    ones = torch.ones_like(y[:,0])
    jac = []
    for j in range(M): 
        grad = torch.autograd.grad(y[:,j], x, grad_outputs=ones, create_graph=True)[0]
        jac.append(grad)
    
    jac = torch.stack(jac, dim=1)
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
    N, K = x.shape
    assert y.shape == (N, K), y.shape
    jac = jacobian(y, x)
    assert jac.shape == (N, K, K)
    div = 0
    for j in range(K):
        div = div + jac[:,j,j]
    return div


def laplacian(y, x):
    '''
    Args:
        y: (N, M) tensor of outputs.
        x: (N, K) tensor of inputs.
    Returns:
        lap: (N, M) Laplacian tensor,
            where lap_ij = sum_k d(dy_ij/dx_ik)_ijk/dx_ik.
    '''
    N, M = y.shape
    N, K = x.shape
    assert y.shape == (N, M)
    jac = jacobian(y, x)
    assert jac.shape == (N, M, K)
    lap = []
    for j in range(M):
        lap.append(divergence(jac[:,j,:], x))
    return torch.stack(lap, dim=1)


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
