import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch import optim
import pandas as pd
import mre_siren


def calc_pde_loss(x, y_pred):
    return torch.zeros(1).cuda()


def train(
    out_prefix,
    data_root,
    downsample,
    batch_size,
    n_hidden,
    n_layers,
    n_epochs,
    lr_init,
    pde_loss_weight,
    print_interval=10,
    save_interval=1000
):
    print('Loading BIOQIC phantom data')
    data = mre_siren.bioqic.BIOQICDataset(
        data_root, downsample=downsample, dtype=torch.float32, device='cuda'
    )
    if batch_size is None or batch_size == 0:
        batch_size = len(data.x)
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

    print('Initializing SIREN model')
    model = mre_siren.models.SIREN(
        n_input=data.x.shape[1],
        n_output=2,
        n_hidden=n_hidden,
        n_layers=n_layers
    ).float().cuda()
    print(model)
    optimizer = optim.Adam(model.parameters(), lr=lr_init)

    metric_index = ['epoch', 'batch']
    metrics = pd.DataFrame(columns=metric_index).set_index(metric_index)

    for i in range(n_epochs+1):
        for j, (x, u) in enumerate(data_loader):
            optimizer.zero_grad()

            u_pred, G_pred = torch.split(model.forward(x), 1, dim=1)

            mse_loss = F.mse_loss(u, u_pred)
            pde_loss = calc_pde_loss(x, u_pred)
            loss = mse_loss + pde_loss_weight * pde_loss

            metrics.loc[(i,j), 'mse_loss'] = mse_loss.item()
            metrics.loc[(i,j), 'pde_loss'] = pde_loss.item()
            metrics.loc[(i,j), 'loss'] = loss.item()

            if i < n_epochs: # don't update on final evaluation
                loss.backward()
                optimizer.step()

        if i % print_interval == 0 or i == n_epochs:
            print(f'[epoch = {i}, batch = {j}] mse_loss = {mse_loss.item():.4f}, pde_loss = {pde_loss.item():.8f}, loss = {loss.item():.4f}')

        if i % save_interval == 0 or i == n_epochs:
            print(f'Saving checkpoint {i}')

            state_file = f'{out_prefix}_{i}.checkpoint'
            torch.save(dict(
                iteration=i, loss=loss,
                model_state=model.state_dict(),
                optim_state=optimizer.state_dict(),
            ), state_file)

            metrics_file = f'{out_prefix}.metrics'
            metrics.to_csv(metrics_file, sep=' ', float_format='%.4f')

    print('Done')


if __name__ == '__main__':
    import fire
    fire.Fire(train)
