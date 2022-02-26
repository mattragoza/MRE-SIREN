import numpy as np
import xarray as xr
import scipy.io
import h5py
import matplotlib as mpl


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
    stored a in dict-like format.
    '''
    for k, v in d.items():
        if hasattr(v, 'shape'):
            print(tab*level + f'{k}: {type(v)} {v.shape} {v.dtype}')
        else:
            print(tab*level + f'{k}: {type(v)}')
        if hasattr(v, 'items'):
            print_mat_info(v, level+1)


def load_bioqic_dataset(data_root, verbose=False):
    '''
    Load the BIOQIC data set of
    phantom MRE images.

    Args:
        data_root: The directory containing
            the MATLAB data files.
        verbose: Print contents of the files.
    Returns:
        An xarray.Dataset of the loaded data.
    '''
    # load the individual matlab files
    data_r = load_mat_data(data_root / 'phantom_raw.mat', verbose)
    data_u = load_mat_data(data_root / 'phantom_unwrapped.mat', verbose)
    data_d = load_mat_data(
        data_root / 'phantom_unwrapped_dejittered.mat', verbose
    )
    data_c = load_mat_data(data_root / 'phantom_raw_complex.mat', verbose)

    # metadata from pdf doc applies to all data
    metadata = dict(
        dims=['freq', 'MEG', 't', 'z', 'x', 'y'],
        coords=dict(
            freq=[30, 40, 50, 60, 70, 80, 90, 100],
            MEG=['y', 'x', 'z'],
            t=np.arange(8) / 8,
            z=np.arange(25) * 0.0015,
            x=np.arange(128) * 0.0015,
            y=np.arange(80) * 0.0015,
        )
    )

    # reverse matlab axes order
    rev_axes = range(6)[::-1]

    # create xarray dataset from downloaded files
    return xr.Dataset(dict(
        magnitude_raw=xr.DataArray(data_r['magnitude'], **metadata),
        magnitude_unwrap=xr.DataArray(data_u['magnitude'], **metadata),
        magnitude_dejit=xr.DataArray(
            data_d['magnitude'].transpose(rev_axes).astype(np.float64),
            **metadata
        ),
        phase_raw=xr.DataArray(data_r['phase'], **metadata),
        phase_unwrap=xr.DataArray(data_u['phase_unwrapped'], **metadata),
        phase_dejit = xr.DataArray(
            data_d['phase_unwrap_noipd'].transpose(range(6)[::-1]),
            **metadata
        ),
        cube_real = xr.DataArray(
            data_c['cube'].real.transpose(rev_axes), **metadata
        ),
        cube_imag = xr.DataArray(
            data_c['cube'].imag.transpose(rev_axes), **metadata
        )
    ))


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


def estimate_phase_shift(a, axis=None):
    '''
    Estimate the global phase shift
    in an MRE phase image using the
    median phase value.

    Args:
        a: Array of MRE phase values.
        axis: Axis or axes along which
            to estimate phase shift(s).
    Returns:
        The estimated phase shift,
        an integer muliple of 2 pi.
    '''
    return np.round(np.median(a, axis=axis) / (2*np.pi))


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
    return hv.Dataset(ds[hue]).to(hv.Image, [x, y], dynamic=True).opts(
        cmap=cmap,
        width=scale*ds.dims[x],
        height=scale*ds.dims[y]
    ).redim.range(**{hue: tuple(v_range)}).hist()
