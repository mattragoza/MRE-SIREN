from torch import optim
import mre_siren


def pde_loss(x, y_pred):
    return 0


def train(
        data_root,
        n_input,
        n_output,
        n_hidden,
        n_layers,
        max_iter,
        lr_init,
        pde_loss_weight,
        print_interval=1000,
        save_interval=1000
    ):
        data = mre-siren.bioqic.BIOQICDataset(data_root)
        x, y = data.as_function(
            var='phase_dejittered', dtype=float, device='cuda'
        )
        model = mre-siren.models.SIREN(n_input, n_output, n_hidden, n_layers)
        model = model.float().cuda()
        optimizer = optim.Adam(model.parameters(), lr=lr_init)
        
        for i in range(max_iter+1):

            y_pred = model.forward(x)
            mse_loss = F.mse_loss(y, y_pred)
            pde_loss = pde_loss(x, y_pred)
            loss = mse_loss + pde_loss_weight * pde_loss

            if i % print_interval == 0:
                print(f'[Iteration {i}] mse_loss = {mse_loss}, ' + 
                    f'pde_loss = {pde_loss}, loss = {loss}')

            if i % save_interval == 0:
                state_file = f'{out_prefix}_{i}.checkpoint'
                torch.save(dict(
                    iteration=i, loss=loss,
                    model_state=model.state_dict(),
                    optim_state=optimizer.state_dict(),
                ), state_file)

            if i == max_iter:
                break

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
