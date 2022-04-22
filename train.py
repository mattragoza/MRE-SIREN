import sys, glob, re
import numpy as np
nax = np.newaxis
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch import optim
import pandas as pd

import mre_siren


def find_last_iter(out_prefix):

    last_iter = -1
    state_pat = re.escape(out_prefix) + r'_(\d+)\.checkpoint'
    state_re = re.compile(state_pat)

    for f in glob.glob(f'{out_prefix}_*.checkpoint'):
        i = int(state_re.match(f).group(1))
        if i > last_iter:
            last_iter = i

    if last_iter < 0:
        raise FileNotFoundError('could not find state file')
    
    return last_iter


def test(out_prefix, data, batch_size, u_model, laplace_dim):

    n_batches = len(data)//batch_size
    print(f'Evaluating test data ({n_batches} batches)')

    x = []
    u_pred = []
    Lu_pred = []
    for batch, (x_, u_, Lu_, m) in enumerate(
        DataLoader(data, batch_size, shuffle=False)
    ):
        # forward pass
        x_.requires_grad = True
        u_pred_ = u_model.forward(x_)
        Lu_pred_ = mre_siren.pde.laplacian(u_pred_, x_, start_dim=laplace_dim)

        x.append(x_.detach())
        u_pred.append(u_pred_.detach())
        Lu_pred.append(Lu_pred_.detach())
        batch += 1

        del x_, u_, Lu_, u_pred_, Lu_pred_, m
        torch.cuda.empty_cache()
        
        print('.', end='', flush=sys.stdout.isatty())
        if batch % 100 == 0:
            print(batch, flush=True)
        elif batch == n_batches:
            print(batch, flush=True)

    # concatenate batches
    x = torch.cat(x, dim=0)
    u_pred = torch.cat(u_pred, dim=0)
    Lu_pred = torch.cat(Lu_pred, dim=0)

    # convert from real and imaginary to complex
    u_pred = u_pred[...,0] + 1j * u_pred[...,1]
    Lu_pred = Lu_pred[...,0] + 1j * Lu_pred[...,1]

    # convert from coordinates to tensors
    u_pred = u_pred.reshape(data.wave.shape).detach()
    Lu_pred = Lu_pred.reshape(data.wave.shape).detach()

    # continuous Laplace inversion
    print('Performing continuous Laplace inversion')
    density = 1000
    frequency = torch.tensor(
        data.ds.coords['frequency'].to_numpy(), device=u_pred.device
    )
    frequency = frequency.reshape([-1] + [1 for i in u_pred.shape[1:]])

    numer_abs_G = density * (2*np.pi * frequency)**2 * torch.abs(u_pred)
    denom_abs_G = torch.abs(Lu_pred)

    numer_cos_G = -(Lu_pred.real * u_pred.real + Lu_pred.imag * u_pred.imag)
    denom_cos_G = torch.abs(Lu_pred) * torch.abs(u_pred)

    dims = (0, -1, -2)
    abs_G = numer_abs_G.sum(dim=dims, keepdims=True) \
        / denom_abs_G.sum(dim=dims, keepdims=True)
    cos_G = numer_cos_G.sum(dim=dims, keepdims=True) \
        / denom_cos_G.sum(dim=dims, keepdims=True)
    phi_G = torch.arccos(cos_G)

    # convert from tensors to numpy arrays
    u_pred = u_pred.cpu().numpy().transpose(data.permutation)
    Lu_pred = Lu_pred.cpu().numpy().transpose(data.permutation)
    abs_G = abs_G.cpu().numpy().transpose(data.permutation)
    phi_G = phi_G.cpu().numpy().transpose(data.permutation)

    # insert numpy arrays into xarray dataset
    abs_G_pred = np.broadcast_to(abs_G, u_pred.shape)
    phi_G_pred = np.broadcast_to(phi_G, u_pred.shape)
    data.ds['wave_shear_pred'] = (data.ds.dims, u_pred)
    data.ds['laplace_wave_shear_pred'] = (data.ds.dims, Lu_pred)
    data.ds['abs_G_pred'] = (data.ds.dims, abs_G_pred)
    data.ds['phi_G_pred'] = (data.ds.dims, phi_G_pred)

    # save to disk
    print('Saving arrays to disk')
    np.save(f'{out_prefix}_wave.nc', u_pred)
    np.save(f'{out_prefix}_laplace.nc', Lu_pred)
    np.save(f'{out_prefix}_abs_G.nc', abs_G_pred)
    np.save(f'{out_prefix}_phi_G.nc', phi_G_pred)


def train(
    out_prefix,
    data_root,
    downsample,
    frequency,
    batch_size,
    n_hidden,
    n_layers,
    sine_w0,
    init_lr,
    n_iters,
    laplace_z=True,
    weight_decay=0,
    print_interval=10,
    test_interval=5000,
    save_interval=5000,
    resume=False,
):
    laplace_z = bool(laplace_z)

    print('Loading training data')
    train_data = mre_siren.bioqic.BIOQICDataset(
        data_root,
        mat_base='phantom_wave_shear.mat',
        phase_shift=True,
        segment=True,
        invert=True,
        invert_kws=dict(use_z=laplace_z),
        make_coords=True,
        downsample=downsample,
        frequency=frequency,
        dtype=torch.float32,
        device='cuda',
        verbose=True
    )

    print('Loading downsampled data')
    down_data = mre_siren.bioqic.BIOQICDataset(
        data_root,
        mat_base='phantom_wave_shear.mat',
        phase_shift=True,
        segment=True,
        invert=True,
        invert_kws=dict(use_z=laplace_z),
        make_coords=True,
        downsample=2*downsample,
        frequency=frequency,
        dtype=torch.float32,
        device='cuda',
        verbose=True
    )

    if batch_size is None or batch_size == 0:
        batch_size = len(train_data.x)
    data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    print('Initializing SIREN model')
    u_model = mre_siren.models.SIREN(
        n_input=train_data.x.shape[1],
        out_shape=train_data.u.shape[1:],
        n_hidden=n_hidden,
        n_layers=n_layers,
        w0=sine_w0,
    ).to(dtype=torch.float32, device='cuda')
    print(u_model)

    x_mean = train_data.x.mean(dim=0)
    x_std = train_data.x.std(dim=0)
    print(f'x_mean = {x_mean}')
    print(f'x_std = {x_std}')

    u_mean = train_data.u.mean(dim=0).reshape(-1)
    u_std = train_data.u.std(dim=0).reshape(-1)
    print(f'u_mean = {u_mean}')
    print(f'u_std = {u_std}')

    u_model.init_weights(
        input_scale=x_std, output_scale=u_std, output_loc=u_mean
    )
    u_optimizer = optim.AdamW(
        u_model.parameters(), lr=init_lr, weight_decay=weight_decay
    )

    metrics_file = f'{out_prefix}.metrics'
    if resume:
        iteration = find_last_iter(out_prefix)
        print(f'Loading checkpoint {iteration}')

        state_file = f'{out_prefix}_{iteration}.checkpoint'
        state = torch.load(state_file)
        u_model.load_state_dict(state['u_model_state'])
        u_optimizer.load_state_dict(state['u_optimizer_state'])

        print(f'Loading {metrics_file}')
        epoch = state['epoch']
        metrics = pd.read_csv(metrics_file, sep=' ')
    else:
        epoch = 0
        iteration = 0
        metrics = pd.DataFrame(columns=['iteration', 'epoch', 'batch', 'freq'])
    
    metrics.set_index(['iteration', 'freq'], inplace=True)
    freqs = train_data.ds.coords['frequency'].to_numpy()
    freq_coords = train_data.x[:,0].unique()
    laplace_dim = 1 if laplace_z else 2

    print('Start training loop')
    while iteration <= n_iters:
        for batch, (x, u, Lu, m) in enumerate(data_loader):

            # only train on segmented region
            x, u, Lu = x[m], u[m], Lu[m]

            # forward pass
            x.requires_grad = True
            u_pred = u_model.forward(x)
            u_mse_loss = F.mse_loss(u, u_pred)

            Lu_pred = mre_siren.pde.laplacian(u_pred, x, start_dim=laplace_dim)
            with torch.no_grad(): # just for monitoring
                Lu_mse_loss = F.mse_loss(Lu, Lu_pred)
            
            # compute and record metrics by frequency
            for freq, freq_coord in zip(freqs, freq_coords):
                assert x.shape[1] == 4, x.shape

                # get frequency mask
                f = (x[:,0] == freq_coord)
                if len(f) == 0:
                    continue

                with torch.no_grad(): # compute metrics
                    f_u_mse_loss = F.mse_loss(u[f], u_pred[f]).item()
                    f_Lu_mse_loss = F.mse_loss(Lu[f], Lu_pred[f]).item()
                    f_u_pred_norm = torch.linalg.norm(
                        torch.flatten(u_pred[f], start_dim=1), dim=1
                    ).mean().item()
                    f_Lu_pred_norm = torch.linalg.norm(
                        torch.flatten(Lu_pred[f], start_dim=1), dim=1
                    ).mean().item()

                # record metrics
                idx = (iteration, freq)
                metrics.loc[idx, 'epoch'] = epoch
                metrics.loc[idx, 'batch'] = batch
                metrics.loc[idx, 'u_mse_loss'] = f_u_mse_loss
                metrics.loc[idx, 'Lu_mse_loss'] = f_Lu_mse_loss
                metrics.loc[idx, 'u_pred_norm'] = f_u_pred_norm
                metrics.loc[idx, 'Lu_pred_norm'] = f_Lu_pred_norm

            if iteration % print_interval == 0 or iteration == n_iters:
                m = metrics.loc[iteration].mean() # mean across frqeuencies
                print(f'[iteration {iteration} epoch {epoch} batch {batch}] u_mse_loss = {m.u_mse_loss:.4e} Lu_mse_loss = {m.Lu_mse_loss:.4e} u_pred_norm = {m.u_pred_norm:.4e} Lu_pred_norm = {m.Lu_pred_norm:.4e}')

                metrics.to_csv(metrics_file, sep=' ', float_format='%.4f')

            if iteration % test_interval == 0 or iteration == n_iters:
                test_prefix = f'{out_prefix}_{iteration}_down'
                test(test_prefix, down_data, batch_size, u_model, laplace_dim)

            if iteration == n_iters:
                break

            # backward pass and update
            u_optimizer.zero_grad()
            u_mse_loss.backward()
            u_optimizer.step()
            iteration += 1
            del x, u, Lu, u_pred, Lu_pred

            if iteration % save_interval == 0 or iteration == n_iters:
                print(f'Saving checkpoint {iteration}')
                state_file = f'{out_prefix}_{iteration}.checkpoint'
                torch.save(dict(
                    epoch=epoch,
                    batch=batch+1,
                    iteration=iteration,
                    u_model_state=u_model.state_dict(),
                    u_optimizer_state=u_optimizer.state_dict()
                ), state_file)

        if iteration == n_iters:
            break

        # end of epoch
        epoch += 1

    del train_data, down_data

    print('Loading full-resolution data')
    full_data = mre_siren.bioqic.BIOQICDataset(
        data_root,
        mat_base='phantom_wave_shear.mat',
        phase_shift=True,
        segment=True,
        invert=True,
        invert_kws=dict(use_z=laplace_z),
        make_coords=True,
        downsample=1,
        frequency=frequency,
        dtype=torch.float32,
        device='cuda',
        verbose=True
    )
    test_prefix = f'{out_prefix}_{iteration}_full'
    test(test_prefix, full_data, batch_size, u_model, laplace_dim)
    del full_data

    if frequency == 'all':

        for upsample in [2, 4]:

            print('Creating super-frequency data')
            super_data = mre_siren.bioqic.BIOQICDataset(
                data_root,
                mat_base='phantom_wave_shear.mat',
                phase_shift=True,
                segment=True,
                invert=True,
                invert_kws=dict(use_z=laplace_z),
                make_coords=True,
                downsample=1,
                upsample=upsample,
                frequency=frequency,
                dtype=torch.float32,
                device='cuda',
                verbose=True
            )
            test_prefix = f'{out_prefix}_{iteration}_super{upsample}'
            test(test_prefix, super_data, batch_size, u_model, laplace_dim)

    print('Done')


if __name__ == '__main__':
    import fire
    fire.Fire(train)
