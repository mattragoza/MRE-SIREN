from pathlib import Path
from functools import lru_cache
import numpy as np
import xarray as xr
import h5py, scipy.io
from scipy.ndimage import gaussian_filter
import torch
import matplotlib as mpl


def ndidx(t):
    '''
    Return a tensor of N-dimensional
    indices for the provided tensor.
    '''
    all_idx = []
    idx = torch.arange(t.numel())
    for size in t.shape[::-1]:
        all_idx.append(torch.remainder(idx, size))
        idx = torch.div(idx, size, rounding_mode='floor')
    return torch.stack(all_idx[::-1], dim=1)


def as_function(t, **kwargs):
    '''
    Represent a tensor as indices and values.

    Args:
        t: An N-dimensional tensor of M values.
    Returns:
        An (M, N) tensor of indices.
        An (M,) tensor of values.
    '''
    t = torch.as_tensor(t, **kwargs)
    idx = torch.as_tensor(ndidx(t), **kwargs)
    return idx, t.reshape(-1)


def as_list(x):
    '''
    Return x in a singleton list if
    it is not already a list object.
    '''
    return x if isinstance(x, list) else [x]


def has(x):
    return x is not None


class Slicer(object):
    '''
    A helper object that allows using
    slicing syntax to get slice objects.
    '''
    def __getitem__(self, idx):
        return idx

s_ = Slicer()


class BIOQICDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        data_root=None,
        select=None,
        downsample=None,
        verbose=False,
        sigma=0.8,
        threshold=120,
        ds=None,
        **kwargs
    ):
        assert has(data_root) ^ has(ds), 'must provide either data_root or ds'

        if data_root: # load MATLAB phantom data files
            self.ds = load_bioqic_dataset(data_root, verbose=verbose)
        else:
            self.ds = ds.copy()

        # subtract estimated phase shifts
        if verbose:
            print(f'Subtracting phase shifts')
        phase = self.ds['phase_unwrap_noipd']
        phase_shift = estimate_phase_shift(phase, axis=(2,3,4,5), keepdims=True)
        if verbose:
            print(phase_shift[...,0,0,0,0])
        phase = phase - phase_shift*2*np.pi
        self.ds['phase_unshifted'] = phase

        # get segmentation mask from magnitude
        if verbose:
            print('Computing segmentation mask')
        magnitude = self.ds['magnitude']
        mask = (gaussian_filter(magnitude, sigma=sigma) > threshold).astype(int)
        self.ds['mask'] = (self.ds.dims, mask)

        if select is not None:
            if verbose:
                print(f'Selecting subset {select}')
            self.ds = self.ds.sel(select)

        if downsample is not None and downsample > 1:
            if verbose:
                print(f'Downsampling by factor {downsample}')
            k = downsample
            self.ds = self.ds.coarsen(x=k, y=k, z=k, boundary='pad').mean()
            self.ds['mask'] = (self.ds['mask'] > 0.5).astype(int)

        if verbose:
            print('Getting numpy arrays')
        self.magnitude = self.ds['magnitude'].to_numpy()
        self.phase = self.ds['phase_unshifted'].to_numpy()
        self.mask = self.ds['mask'].to_numpy()

        # convert to function representation
        if verbose:
            print('Getting indices and values')
        self.x, self.u = as_function(self.phase, **kwargs)
        self.m = (self.mask.reshape(-1) > 0)
        self.x.requires_grad = True

        if verbose:
            print(self.phase.shape, self.x.shape, self.u.shape)

    def view(self, var, scale=1.0, v_min=0, v_max=None):
        import holoviews as hv
        return hv.Layout([
            view_xarray(
                self.ds, hue=h, x='x', y='y',
                cmap=phase_color_map() if 'phase' in h else magnitude_color_map(),
                v_range=(-2*np.pi, 2*np.pi) if 'phase' in h else (v_min, v_max),
                scale=scale
            ) for h in as_list(var)
        ])

    def __getitem__(self, idx):
        return self.x[idx], self.u[idx], self.m[idx]

    def __len__(self):
        return len(self.x)


@lru_cache(4)
def load_mat_data(mat_file, verbose=False):
    '''
    Load data file from MATLAB file.

    Args:
        mat_file: Filename, typically .mat.
        verbose: Print some info about the
            contents of the file.
    Returns:
        Loaded data in a dict-like format.
    '''
    mat_file = str(mat_file)
    try:
        data = scipy.io.loadmat(mat_file)
    except NotImplementedError as e:
        # Please use HDF reader for matlab v7.3 files
        data = h5py.File(mat_file)
    if verbose:
        print(mat_file)
        print_mat_info(data, level=1)
    return data


def print_mat_info(d, level=0, tab=' '*4):
    '''
    Recursively print information
    about the contents of a data set
    stored in dict-like format.
    '''
    for k, v in d.items():
        if hasattr(v, 'shape'):
            print(tab*level + f'{k}: {type(v)} {v.shape} {v.dtype}')
        else:
            print(tab*level + f'{k}: {type(v)}')
        if hasattr(v, 'items'):
            print_mat_info(v, level+1)


def load_bioqic_dataset(data_root, all=False, verbose=False):
    '''
    Load the BIOQIC phantom MRE data set.

    Args:
        data_root: The directory containing
            the MATLAB data files.
        all: Load all .mat files instead of just
            phantom_unwrapped_dejittered.mat.
        verbose: Print contents of the files.
    Returns:
        An xarray.Dataset of the loaded data.
    '''
    data_root = Path(data_root)

    # metadata from pdf doc applies to all data
    metadata = dict(
        dims=['freq', 'MEG', 't', 'z', 'x', 'y'],
        coords=dict(
            freq=[30, 40, 50, 60, 70, 80, 90, 100],
            MEG=['y', 'x', 'z'],
            t=np.arange(  8) / 8,
            z=np.arange( 25) * 0.0015,
            x=np.arange(128) * 0.0015,
            y=np.arange( 80) * 0.0015,
        )
    )

    # specify the vars that we will get from each file
    # we also need to reverse axes order for some files
    data_vars = [
        ('phantom_raw_complex.mat', ['cube'], np.complex128, True),
        ('phantom_raw.mat',         ['phase'], np.float64, False),
        ('phantom_unwrapped.mat',   ['phase_unwrapped'], np.float64, False),
        (
            'phantom_unwrapped_dejittered.mat',
                ['phase_unwrap_noipd', 'magnitude'], np.float64, True
        )
    ]
    if not all: # just load the processed phase and magnitude data
        data_vars = data_vars[-1:]

    # load the individual matlab files into xarrays
    xr_data = dict()
    for mat_base, mat_vars, dtype, rev_axes in data_vars:
        mat_data = load_mat_data(data_root/mat_base, verbose)
        for var in mat_vars:
            arr = mat_data[var].astype(dtype)
            if rev_axes:
                arr = arr.transpose(range(6)[::-1])
            xr_data[var] = xr.DataArray(arr, **metadata)

    # create dataset from xarrays
    return xr.Dataset(xr_data)


def magnitude_color_map():
    '''
    Create a linear grayscale colormap.
    '''
    colors = [(0,0,0), (1,1,1)]
    return mpl.colors.LinearSegmentedColormap.from_list(
        name='magnitude', colors=colors, N=255
    )


def phase_color_map():
    '''
    Create a colormap for MRE phase images
    from yellow, red, black, blue, to cyan.
    '''
    colors = [(1,1,0), (1,0,0), (0,0,0), (0,0,1), (0,1,1)]
    return mpl.colors.LinearSegmentedColormap.from_list(
        name='phase', colors=colors, N=255
    )


def view_xarray(ds, x, y, hue, cmap, v_range, scale):
    '''
    Interactively view an xarray
    defined in an xarray Dataset.

    NOTE that each dim must have
    numeric coordinates defined.
    
    Args:
        ds: An xarray Dataset.
        x: The x dimension.
        y: The y dimension.
        hue: The data to view.
        cmap: Color map object.
        v_range: Color map value range.
        scale: Figure scale.
    Returns:
        A holoviews Image object.
    '''
    import holoviews as hv
    v_min, v_max = v_range
    if v_min is None:
        v_min = ds[hue].min()
    if v_max is None:
        v_max = ds[hue].max()
    return hv.Dataset(ds[hue]).to(hv.Image, [x, y], dynamic=True).opts(
        cmap=cmap,
        width=scale*ds.dims[x],
        height=scale*ds.dims[y]
    ).redim.range(**{hue: tuple(v_range)}).hist()


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
