import sys
import numpy as np
nax = np.newaxis
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch import optim
import pandas as pd

import mre_siren


def test(out_prefix, data, batch_size, u_model):

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
        Lu_pred_ = mre_siren.pde.laplacian(u_pred_, x_, start_dim=1)

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
    print('Continuous Laplace inversion')
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
    weight_decay=0,
    print_interval=10,
    test_interval=5000,
    save_interval=5000
):
    print('Loading training data')
    train_data = mre_siren.bioqic.BIOQICDataset(
        data_root,
        mat_base='phantom_wave_shear.mat',
        phase_shift=True,
        segment=True,
        invert=True,
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

    columns = ['iteration', 'epoch', 'batch']
    metrics = pd.DataFrame(columns=columns).set_index('iteration')

    print('Start training loop')
    epoch = 0
    iteration = 0
    while True:
        for batch, (x, u, Lu, m) in enumerate(data_loader):

            # only train on segmented region
            x, u, Lu = x[m], u[m], Lu[m]

            # forward pass
            x.requires_grad = True
            u_pred = u_model.forward(x)
            u_mse_loss = F.mse_loss(u, u_pred)

            Lu_pred = mre_siren.pde.laplacian(u_pred, x, start_dim=1)
            with torch.no_grad(): # just for monitoring
                Lu_mse_loss = F.mse_loss(Lu, Lu_pred)
            
            # record loss metrics
            metrics.loc[iteration, 'epoch'] = epoch
            metrics.loc[iteration, 'batch'] = batch
            metrics.loc[iteration, 'u_mse_loss'] = u_mse_loss.item()
            metrics.loc[iteration, 'Lu_mse_loss'] = Lu_mse_loss.item()
            metrics.loc[iteration, 'u_pred_norm'] = torch.linalg.norm(
                torch.flatten(u_pred, start_dim=1), dim=1
            ).mean().item()
            metrics.loc[iteration, 'Lu_pred_norm'] = torch.linalg.norm(
                torch.flatten(Lu_pred, start_dim=1), dim=1
            ).mean().item()

            if iteration % print_interval == 0 or iteration == n_iters:
                m = metrics.loc[iteration]
                print(f'[iteration {iteration} epoch {epoch} batch {batch}] u_mse_loss = {m.u_mse_loss:.4e} Lu_mse_loss = {m.Lu_mse_loss:.4e} u_pred_norm = {m.u_pred_norm:.4e} Lu_pred_norm = {m.Lu_pred_norm:.4e}')

                metrics_file = f'{out_prefix}.metrics'
                metrics.to_csv(metrics_file, sep=' ', float_format='%.4f')

            if iteration % test_interval == 0 or iteration == n_iters:
                test_prefix = f'{out_prefix}_{iteration}_down'
                test(test_prefix, down_data, batch_size, u_model)

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
        make_coords=True,
        downsample=1,
        frequency=frequency,
        dtype=torch.float32,
        device='cuda',
        verbose=True
    )
    test_prefix = f'{out_prefix}_{iteration}_full'
    test(test_prefix, full_data, batch_size, u_model)

    print('Done')


if __name__ == '__main__':
    import fire
    fire.Fire(train)
