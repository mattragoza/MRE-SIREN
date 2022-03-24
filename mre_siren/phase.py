import numpy as np
from scipy.ndimage import gaussian_filter


def estimate_phase_shift(a, **kwargs):
    '''
    Estimate the global phase shift
    in an MRE phase image using the
    median phase value.

    Args:
        a: Array of MRE phase values.
        kwargs: Keyword arguments passed
            to numpy.median.
    Returns:
        The estimated phase shift(s),
            an integer muliple of 2 pi.
    '''
    return np.round(np.median(a, **kwargs) / (2*np.pi))


def subtract_phase_shift(a, **kwargs):
    return a - estimate_phase_shift(a, **kwargs) * 2*np.pi


def phase_to_complex(phase, conj=False):
    '''
    Convert phase to complex value.
    This is the inverse of np.angle,
    up to an integer multiple of 2 pi.
    '''
    if conj:
        return np.exp(-1j*phase)
    else:
        return np.exp(1j*phase)


def smooth_phase(phase, sigma=0.65, truncate=3, use_z=True):
    '''
    Apply Gaussian smoothing to phase
    in the complex domain.
    '''
    n_freq, n_MEG, n_t, n_z, n_x, n_y = phase.shape
    smoothed_phase = phase.copy()
    
    # iterate over 2D slices or 3D volumes
    for f in range(n_freq):
        for m in range(n_MEG):
            for t in range(n_t):
                
                if use_z:
                    p = phase_to_complex(phase[f,m,t])
                    p = gaussian_filter(p, sigma=sigma, truncate=truncate)
                    p = np.angle(p)
                    smoothed_phase[f,m,t] = p
                else:
                    for z in range(n_z):
                        p = phase_to_complex(phase[f,m,t,z])
                        p = gaussian_filter(p, sigma=sigma, truncate=truncate)
                        p = np.angle(p)
                        smoothed_phase[f,m,t,z] = p
    
    return smoothed_phase


def grad_wrap_FFT(phase, resolution=0.0015, n_harmonics=1):

    # get the dimensions
    n_freq, n_MEG, n_t, n_z, n_x, n_y = phase.shape

    # loop for unwrapping and fft
    wave_field = np.zeros(
        (n_freq, n_MEG, 2, n_z, n_x, n_y), dtype=np.complex128
    )
    
    n_dims = 2
    grad = np.zeros((n_dims, n_t, n_z, n_x, n_y), dtype=np.complex128)
    
    for f in range(n_freq):
        for m in range(n_MEG):
            for t in range(n_t):
                
                p = phase[f,m,t,:,:,:]
                p_comp = phase_to_complex(p)
                p_conj = phase_to_complex(p, conj=True)
                
                # compute spatial gradient
                p_z, p_x, p_y = np.gradient(p_comp, resolution)
                
                # in-plane derivative components
                # NOTE: it's VERY important to multiply by p_conj here
                #  instead of p or p_comp. I don't fully understand why.
                grad[0,t] = (p_y * p_conj).imag
                grad[1,t] = (p_x * p_conj).imag
            
            # fourier transformation and selection of harmonic
            fourier1 = np.fft.fft(grad[0], axis=0);
            wave_field[f,m,0,:,:,:] = fourier1[n_harmonics,:,:,:]
            
            fourier2 = np.fft.fft(grad[1], axis=0);
            wave_field[f,m,1,:,:,:] = fourier2[n_harmonics,:,:,:]
            
    return wave_field


def calc_k_filter(n, resolution):
    # I verified this against MATLAB code
    return -(np.arange(n) - np.fix(n/2)) / (n * resolution)


def radial_filter(wave, resolution=0.0015, threshold=100, order=1, use_z=False):
    
    n_freq, n_MEG, n_grad, n_z, n_x, n_y = wave.shape

    # create k-space filter (spatial frequency space)
    nax = np.newaxis
    if use_z:
        k_z = calc_k_filter(n_z, resolution)
        k_x = calc_k_filter(n_x, resolution)
        k_y = calc_k_filter(n_y, resolution)
        abs_k = np.sqrt(
            np.abs(k_z[:,nax,nax])**2 +
            np.abs(k_x[nax,:,nax])**2 +
            np.abs(k_y[nax,nax,:])**2
        )
    else:
        k_x = calc_k_filter(n_x, resolution)
        k_y = calc_k_filter(n_y, resolution)
        abs_k = np.sqrt(np.abs(k_x[:,nax])**2 + np.abs(k_y[nax,:])**2)
    
    k_filter = 1 / (1 + (abs_k/threshold)**(2*order))
    k_filter = np.fft.ifftshift(k_filter)

    shear_wave = wave.copy()
   
    # iterate over 3D volumes
    for f in range(n_freq):
        for m in range(n_MEG):
            for g in range(n_grad):

                if use_z:
                    u = wave[f,m,g]

                    # this is applying a convolution
                    u = np.fft.fftn(u)
                    u = u * k_filter
                    u = np.fft.ifftn(u)

                    shear_wave[f,m,g] = u
                    continue
                
                for z in range(n_z):
                    u = wave[f,m,g,z]

                    # this is applying a convolution
                    u = np.fft.fftn(u)
                    u = u * k_filter
                    u = np.fft.ifftn(u)

                    shear_wave[f,m,g,z] = u
    
    return shear_wave


def laplace_inversion(
    wave, frequency, resolution=0.0015, density=1000, eps=1e-8
):
    n_freq, n_MEG, n_grad, n_z, n_x, n_y = wave.shape
    
    strain = wave[0,0,0].real * 0
    numer_phi = strain.copy()
    denom_phi = strain.copy()
    numer_absG = strain.copy()
    denom_absG = strain.copy()
   
    # iterate over 3D volumes
    for f in range(n_freq):
        for m in range(n_MEG):
            for g in range(n_grad):

                u = wave[f,m,g]
                u_z,  u_x,  u_y  = np.gradient(u,   resolution)
                u_zz, u_zx, u_zy = np.gradient(u_z, resolution)
                u_xz, u_xx, u_xy = np.gradient(u_x, resolution)
                u_yz, u_yx, u_yy = np.gradient(u_y, resolution)
                laplace_u = u_xx + u_yy #+ u_zz
                
                numer_phi += laplace_u.real * u.real + laplace_u.imag * u.imag
                denom_phi += np.abs(laplace_u) * np.abs(u)
                
                numer_absG += density * (2*np.pi * frequency[f])**2 * np.abs(u)
                denom_absG += np.abs(laplace_u)
                
                strain += np.abs(u)
    
    phi = np.arccos(-numer_phi/(denom_phi + eps))
    absG = numer_absG / (denom_absG + eps)

    return absG, phi, strain


def laplace_invert(
    u, frequency, resolution=0.0015, density=1000, eps=1e-8
):
    # assume last three dimensions are z, x, y
    u_z,  u_x,  u_y  = np.gradient(u,   resolution, axis=(-3,-2,-1))
    u_zz, u_zx, u_zy = np.gradient(u_z, resolution, axis=(-3,-2,-1))
    u_xz, u_xx, u_xy = np.gradient(u_x, resolution, axis=(-3,-2,-1))
    u_yz, u_yx, u_yy = np.gradient(u_y, resolution, axis=(-3,-2,-1))
    laplace_u = u_xx + u_yy #+ u_zz

    # solve for the complex shear modulus
    numer_abs_G = density * (2*np.pi * frequency)**2 * np.abs(u)
    denom_abs_G = np.abs(laplace_u)

    numer_cos_G = -(laplace_u.real * u.real + laplace_u.imag * u.imag)
    denom_cos_G = np.abs(laplace_u) * np.abs(u)
    
    axes = tuple(range(u.ndim-3))
    abs_G = numer_abs_G.sum(axis=axes) / denom_abs_G.sum(axis=axes)
    cos_G = numer_cos_G.sum(axis=axes) / denom_cos_G.sum(axis=axes)
    phi_G = np.arccos(cos_G)

    return laplace_u, abs_G, phi_G
