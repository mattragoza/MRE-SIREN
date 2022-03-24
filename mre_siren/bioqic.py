import sys
from pathlib import Path
from functools import lru_cache
import numpy as np
import xarray as xr
import h5py, scipy.io
from scipy.ndimage import gaussian_filter
import torch
import matplotlib as mpl

from . import phase


def nd_coords(shape, resolution=1, center=False):
    '''
    Return a tensor of N-dimensional
    coordinates for a grid with the
    provided shape (M total values).

    Args:
        shape: The shape of the N-dimensional grid.
        resolution: Spatial resolution of grid dimensions.
        center: Subtract mean from spatial coordinates.
    Returns:
        An (M, N) tensor of coordinates.
    '''
    n_dims = len(shape)

    # for each dim, get d points spaced according to resolution
    resolution = np.broadcast_to(resolution, (n_dims,))
    dims = [torch.arange(d)*r for d,r in zip(shape, resolution)]

    # get n-demsenional coordinates from dims
    coords = torch.cartesian_prod(*dims)

    if center: # subtract the center
        coords -= coords.mean(dim=0)

    return coords


def as_nd_coords(a, n=3, resolution=1, center=False, **kwargs):
    '''
    Represent an N-dimensional array as 
    tensors of spatial coordinates and values.

    Args:
        a: An N-dimensional array of M values.
        n: The spatial dimensionality N, assuming
            that these are the first N dimensions.
        resolution: Resolution of each dimension.
        center: Subtract mean of coordinates.
        **kwargs: Passed to torch.as_tensor.
    Returns:
        An (M, N) tensor of coordinates.
        An (M, D) tensor of values, where
            D = 2 if t is complex, else D = 1.
    '''
    # get coordinates and values
    coords = nd_coords(a.shape[:n], resolution[:n], center)
    values = a.reshape(-1, *a.shape[n:])

    if np.iscomplexobj(values):
        values = np.stack([values.real, values.imag], axis=-1)

    return (
        torch.as_tensor(coords, **kwargs),
        torch.as_tensor(values, **kwargs)
    )


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
        mat_base='phantom_wave_shear.mat',
        phase_shift=False,
        segment=True,
        invert=False,
        make_coords=True,
        select=None,
        upsample=None,
        downsample=None,
        verbose=False,
        sigma=0.8,
        threshold=120,
        ds=None,
        wave_var=None,
        **kwargs
    ):
        assert has(data_root) ^ has(ds), 'must provide either data_root or ds'

        if data_root: # load MATLAB phantom data files
            print('Loading data set from disk')
            self.ds, wave_var = load_bioqic_dataset(
                data_root, mat_base, verbose=verbose
            )
        else:
            print('Using provided data set')
            self.ds = ds.copy()
            assert wave_var, 'must specify wave variable'

        self.wave_var = wave_var
        if verbose:
            print(f'Wave variable is {wave_var} {self.ds[wave_var].shape}')

        if phase_shift: # subtract estimated phase shifts
            if verbose:
                print(f'Subtracting estimated phase shifts')
            axes = [i for i,d in enumerate(self.ds.dims) if d in 'txyz']
            self.ds[wave_var] = phase.subtract_phase_shift(
                self.ds[wave_var], axis=axes, keepdims=True
            )

        if segment: # get segmentation mask from magnitude
            if verbose:
                print('Computing segmentation mask from magnitude')
            mask = (
                gaussian_filter(self.ds['magnitude'], sigma=sigma) > threshold
            ).astype(int)
            self.ds['mask'] = (self.ds.dims, mask)

        if select is not None:
            if verbose:
                print(f'Selecting data subset {select}')
            self.ds = self.ds.sel(select)

        # temporal upsampling
        if upsample is not None and upsample > 1:
            if verbose:
                print(f'Temporal upsampling by factor {upsample}')
            t = nd_coords(
                shape=(upsample*self.ds.dims['t'],),
                extent=1,
                center=False
            )
            self.ds = self.ds.interp(t=t)

        # spatial downsampling
        if downsample is not None and downsample > 1:
            if verbose:
                print(f'Spatial downsampling by factor {downsample}')
            k = downsample
            self.ds = self.ds.coarsen(x=k, y=k, z=k, boundary='pad').mean()
            if segment:
                self.ds['mask'] = (self.ds['mask'] > 0.5).astype(int)

        if invert: # perform Laplace inversion
            if verbose:
                print('Performing discrete Laplace inversion')
            laplace_wave, abs_G, phi_G = phase.laplace_invert(
                self.ds[self.wave_var], self.ds.coords['frequency']
            )
            self.ds[f'laplace_{self.wave_var}'] = (self.ds.dims, laplace_wave)
            self.ds['abs_G'] = abs_G
            self.ds['phi_G'] = phi_G

        # reorder the columns
        for var in list(self.ds.keys()):
            if var not in {'magnitude', 'mask'}:
                temp = self.ds[var]
                del self.ds[var]
                self.ds[var] = temp

        if verbose:
            print('Getting numpy arrays')

        self.magnitude = self.ds['magnitude'].to_numpy()
        self.wave = self.ds[wave_var].to_numpy()
        if segment:
            self.mask = self.ds['mask'].to_numpy()

        if verbose:
            print(self.wave.shape)

        if make_coords: # convert to coordinate representation
            if verbose:
                print('Getting spatial coordinate representation')

            self.resolution = get_xarray_resolution(self.ds)
            if verbose:
                print(f'resolution = {self.resolution}')

            # move spatial dims to front
            n_dims = self.wave.ndim
            spatial_dims = [-2, -1, -3]
            spatial_dims = [
                d%n_dims for d in spatial_dims
            ]
            self.permutation = spatial_dims + [
                d for d in range(n_dims) if d not in spatial_dims
            ]
            if verbose:
                print(f'permutation = {self.permutation}')

            self.magnitude = self.magnitude.transpose(self.permutation)
            self.wave = self.wave.transpose(self.permutation)
            if segment:
                self.mask = self.mask.transpose(self.permutation)
            self.resolution = self.resolution[self.permutation]

            # get the nd coordinates and values
            self.x, self.u, = as_nd_coords(
                self.wave, resolution=self.resolution, center=True, **kwargs
            )
            if segment:
                mask = self.mask.reshape(-1, *self.mask.shape[3:])
                mask = self.mask.reshape(mask.shape[0], -1)
                self.m = (mask > 0).all(axis=1)

            if verbose:
                print(f'x shape = {self.x.shape}\nu shape = {self.u.shape}')

        if verbose:
            print(f'Data set initialized')

    def view(
        self, var=None, scale=4, share=False, pct=2.5, n_cols=2, verbose=True
    ):
        import holoviews as hv

        if var is None: # view all variables
            var = list(self.ds.keys())

        if verbose:
            print(f'Viewing {var}')

        hv_images = []
        for v in as_list(var):

            if 'wave' in v or 'phase' in v:
                cmap = phase_color_map()
            else:
                cmap = magnitude_color_map()
            v_min = v_max = None

            if np.iscomplexobj(self.ds[v]):
                if 'elast' in v:
                    funcs = [np.absolute, xr.ufuncs.angle]
                    cmap = [magnitude_color_map(), magnitude_color_map()]
                    share_vlim = False
                else:
                    funcs = [np.real, np.imag]
                    cmap = [cmap, cmap]
                    share_vlim = True

                hv_images.append(view_xarray(
                    self.ds, var=v, x='x', y='y', func=funcs[0],
                    cmap=cmap[0], v_min=v_min, v_max=v_max, scale=scale,
                    share=share, pct=pct, verbose=verbose, share_vlim=share_vlim
                ))
                hv_images.append(view_xarray(
                    self.ds, var=v, x='x', y='y', func=funcs[1],
                    cmap=cmap[1], v_min=v_min, v_max=v_max, scale=scale,
                    share=share, pct=pct, verbose=verbose, share_vlim=share_vlim
                ))
            else:
                hv_images.append(view_xarray(
                    self.ds, var=v, x='x', y='y', func=None,
                    cmap=cmap, v_min=v_min, v_max=v_max, scale=scale,
                    share=share, pct=pct, verbose=verbose
                ))

        return hv.Layout(hv_images).cols(n_cols)

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
        Flag indicating MATLAB axes order.
            (i.e. if True, then reverse order)
    '''
    mat_file = str(mat_file)
    try:
        data = scipy.io.loadmat(mat_file)
        rev_axes = True
    except NotImplementedError as e:
        # Please use HDF reader for matlab v7.3 files
        data = h5py.File(mat_file)
        rev_axes = False
    except:
        print(mat_file, file=sys.stderr)
        raise
    if verbose:
        print(mat_file)
        print_mat_info(data, level=1)
    return data, rev_axes


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


def load_bioqic_dataset(data_root, mat_base, verbose=False):
    '''
    Load the BIOQIC phantom MRE data set.

    Args:
        data_root: The directory containing
            the MATLAB data files.
        which: Which MATLAB data file(s) to load.
        all: Load all .mat files instead of just
            phantom_unwrapped_dejittered.mat.
        verbose: Print contents of the files.
    Returns:
        An xarray.Dataset of the loaded data.
        Name of most-processed wave variable
            loaded from the selected files.
    '''
    data_root = Path(data_root)

    # metadata from pdf document
    metadata = dict(
        dims=['frequency', 'MEG', 't', 'z', 'x', 'y'],
        coords=dict(
            frequency=[30, 40, 50, 60, 70, 80, 90, 100],
            MEG=['z', 'x', 'y'],
            t=np.arange(8) / 8,
            z=np.arange(25)  * 0.0015,
            x=np.arange(128) * 0.0015,
            y=np.arange(80)  * 0.0015,
        )
    )

    # variables defined in each matlab file
    data_vars = {
        'phantom_raw_complex.mat': ['cube'],
        'phantom_raw.mat': ['magnitude', 'phase'],
        'phantom_unwrapped.mat': ['magnitude', 'phase_unwrapped'],
        'phantom_unwrapped_dejittered.mat': ['magnitude', 'phase_unwrap_noipd'],
        'phantom_smoothed.mat': ['magnitude', 'phase_smoothed'],
        'phantom_wave.mat': ['magnitude', 'wave'],
        'phantom_wave_shear.mat': ['magnitude', 'wave_shear'], 
        'phantom_MDEV.mat': ['magnitude', 'absG', 'phi', 'strain']
    }

    # put the data into xarrays
    xr_data = dict()
    for mat_base in as_list(mat_base):

        # load the selected matlab file
        mat_data, rev_axes = load_mat_data(data_root/mat_base, verbose)

        # for each variable in the file,
        for var in data_vars[mat_base]:

            # get numpy array from matlab data
            arr = np.array(mat_data[var])
            arr_metadata = metadata.copy()
            n_dims = arr.ndim

            if rev_axes: # some matlab files use reversed order
                arr = arr.transpose(range(n_dims)[::-1])

            if n_dims < 6: # use only the relevant metadata
                arr_metadata['dims'] = metadata['dims'][-n_dims:]
                arr_metadata['coords'] = {
                    k: metadata['coords'][k] for k in arr_metadata['dims']
                }

            if 'wave' in mat_base: # after Fourier transform
                arr_metadata['coords']['t'] = range(arr.shape[2])

            xr_data[var] = xr.DataArray(arr, **arr_metadata)

    return xr.Dataset(xr_data), var


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
    Create a colormap for MRE wave images
    from yellow, red, black, blue, to cyan.
    '''
    colors = [(1,1,0), (1,0,0), (0,0,0), (0,0,1), (0,1,1)]
    return mpl.colors.LinearSegmentedColormap.from_list(
        name='wave', colors=colors, N=255
    )


def view_xarray(
    ds, x, y, var, cmap,
    v_min=None, v_max=None, scale=1, func=None, share=True, pct=2.5,
    verbose=False, share_vlim=True
):
    '''
    Interactively view an xarray
    defined in an xarray Dataset.

    NOTE that each dim must have
    numeric coordinates defined.
    
    Args:
        ds: An xarray Dataset.
        x: The x dimension.
        y: The y dimension.
        var: The data to view.
        cmap: Color map object.
        v_min: Minimum color value,
            or None to infer.
        v_max: Maximum color value,
            or None to infer.
        scale: Figure scale.
        func: View function of data.
        share: Infer v_range from reference data,
            if var starts with my_ or pred_.
        pct: Percentile for inferring v_range.
    Returns:
        A holoviews Image object.
    '''
    import holoviews as hv

    data = ds[var]
    var_label = var

    ref_var = None
    if share: # determine reference variable
        if var.startswith('my_'):
            ref_var = var[3:]
        elif var.endswith('_pred'):
            ref_var = var[:-5]
    if ref_var:
        ref_data = ds[ref_var]

    if func is not None: # apply function to data
        data = func(data)
        var_label = f'{func.__name__}({var})'
        if ref_var:
            ref_data = func(ref_data)

    # infer value range from data or reference data
    if v_min is None:
        v_min = np.percentile(ref_data if ref_var else data, pct)
    if v_max is None:
        v_max = np.percentile(ref_data if ref_var else data, 100-pct)

    print(var, ref_var, func, v_min, v_max)

    image = hv.Dataset(data).to(hv.Image, [x, y], dynamic=True)
    image = image.opts(
        cmap=cmap, width=scale*ds.dims[x], height=scale*ds.dims[y]
    )
    image = image.redim.label(**{var: var_label})
    image = image.hist().redim.label(**{var: var_label})
    if share_vlim:
        return image.redim.range(**{var: (v_min, v_max)})
    else:
        return image.opts(
            hv.opts.Image(clim=(v_min, v_max)),
            hv.opts.Histogram(xlim=(v_min, v_max), axiswise=True)
        )


def get_xarray_resolution(xr):
    '''
    Get the resolutions of an xarray
    object, assuming uniform spacing
    along each dimension.

    Args:
        xr: N-dimensional xr.DataArray
            or xr.Dataset object.
    Returns:
        N vector of resolutions.
    '''
    resolution = []
    for k, v in xr.coords.items():
        try:
            r = (v[1] - v[0]).to_numpy()
            resolution.append(r)
        except TypeError: # dim has non-numeric coords
            resolution.append(1)
        except IndexError: # dim has length 1
            continue
    return np.array(resolution)
