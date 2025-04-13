import torch
import numpy as np


def plot_two_d_vector_field(dynamics_fn, axs, min_xy=-3, max_xy=3, n_pts=500, device='cpu', plot_args=None, alpha=1.):
    if plot_args:
        with torch.no_grad():
            x = np.linspace(plot_args['min_x'], plot_args['max_x'], plot_args['n_pts'])
            y = np.linspace(plot_args['min_y'], plot_args['max_y'], plot_args['n_pts'])
            X, Y = np.meshgrid(x, y)

            XY = torch.zeros((X.shape[0] ** 2, 2), device=device)
            XY[:, 0] = torch.from_numpy(X).flatten().to(device)
            XY[:, 1] = torch.from_numpy(Y).flatten().to(device)

            XY_out = dynamics_fn(XY)
            s = XY_out - XY
            u = s[:, 0].reshape(X.shape[0], X.shape[1])
            v = s[:, 1].reshape(Y.shape[0], Y.shape[1])

            axs.streamplot(X, Y, u.cpu(), v.cpu(), color='black',
                           linewidth=plot_args['linewidth'],
                           arrowsize=plot_args['arrowsize'],
                           density=plot_args['density'])
    else:
        with torch.no_grad():
            x = np.linspace(min_xy, max_xy, n_pts)
            y = np.linspace(min_xy, max_xy, n_pts)
            X, Y = np.meshgrid(x, y)

            XY = torch.zeros((X.shape[0]**2, 2), device=device)
            XY[:, 0] = torch.from_numpy(X).flatten().to(device)
            XY[:, 1] = torch.from_numpy(Y).flatten().to(device)

            XY_out = dynamics_fn(XY)
            s = XY_out - XY
            u = s[:, 0].reshape(X.shape[0], X.shape[1])
            v = s[:, 1].reshape(Y.shape[0], Y.shape[1])

            stream = axs.streamplot(X, Y, u.cpu(), v.cpu(), color='black', linewidth=0.5, arrowsize=0.25, density=1.)

        return stream


def plot_three_d_vector_field(dynamics_fn, axs, min_xy=-3, max_xy=3, n_pts=100, device='cpu'):
    with torch.no_grad():
        x = np.linspace(min_xy, max_xy, n_pts)
        y = np.linspace(min_xy, max_xy, n_pts)
        z = np.linspace(min_xy, max_xy, 4)
        X, Y, Z = np.meshgrid(x, y, z)

        grid_points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
        grid_points_tensor = torch.tensor(grid_points, dtype=torch.float32, device=device)

        with torch.no_grad():
            vector_field_tensor = dynamics_fn(grid_points_tensor)
            vector_field = vector_field_tensor.cpu().numpy()

        # Reshape back to grid shape
        U = vector_field[:, 0].reshape(X.shape)
        V = vector_field[:, 1].reshape(Y.shape)
        W = vector_field[:, 2].reshape(Z.shape)

        axs.quiver(X, Y, Z, U, V, W, length=10., normalize=True, color='b', alpha=0.5)


def plot_two_d_vector_field_w_input(dynamics_fn, input, axs, min_xy=-3, max_xy=3, n_pts=500, device='cpu'):
    with torch.no_grad():
        x = np.linspace(min_xy, max_xy, n_pts)
        y = np.linspace(min_xy, max_xy, n_pts)
        X, Y = np.meshgrid(x, y)

        XY = torch.zeros((X.shape[0]**2, 2), device=device)
        XY[:, 0] = torch.from_numpy(X).flatten().to(device)
        XY[:, 1] = torch.from_numpy(Y).flatten().to(device)

        XY_out = dynamics_fn(XY + input)
        s = XY_out - XY
        u = s[:, 0].reshape(X.shape[0], X.shape[1])
        v = s[:, 1].reshape(Y.shape[0], Y.shape[1])

        axs.streamplot(X, Y, u.cpu(), v.cpu(), color='black', linewidth=0.5, arrowsize=0.5)


def plot_three_d_vector_field_w_input(dynamics_fn, input, axs, min_xy=-3, max_xy=3, n_pts=500, device='cpu'):
    with torch.no_grad():
        x = np.linspace(min_xy, max_xy, n_pts)
        y = np.linspace(min_xy, max_xy, n_pts)
        z = np.linspace(min_xy, max_xy, 4)
        X, Y, Z = np.meshgrid(x, y, z)

        grid_points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
        grid_points_tensor = torch.tensor(grid_points, dtype=torch.float32, device=device)

        with torch.no_grad():
            vector_field_tensor = dynamics_fn(grid_points_tensor)
            vector_field = vector_field_tensor.cpu().numpy()

        # Reshape back to grid shape
        U = vector_field[:, 0].reshape(X.shape)
        V = vector_field[:, 1].reshape(Y.shape)
        W = vector_field[:, 2].reshape(Z.shape)

        axs.quiver(X, Y, Z, U, V, W, length=0.1, normalize=True, color='b', alpha=0.5)


def plot_spikes(spikes, axs):
    n_bins = spikes.shape[0]
    n_neurons = spikes.shape[1]

    # fig, axs = plt.subplots(figsize=(6, 3))
    _, indices = torch.sort(spikes.mean(dim=0))
    spikes = spikes[:, indices][..., n_neurons//2:]
    n_neurons = n_neurons//2

    for n in range(n_neurons):
        time_ax = np.arange(n_bins)
        neuron_spikes = spikes[:, n]
        neuron_spikes[neuron_spikes > 0] = 1
        neuron_spikes = neuron_spikes * time_ax
        neuron_spikes = neuron_spikes[neuron_spikes > 0]

        axs.scatter(neuron_spikes, 0.5 * n * np.ones_like(neuron_spikes), marker='o', color='black', s=4,
                    edgecolors='none')


def generate_gray_gradient(N):
    # Define the RGB values for light-gray and dark-gray
    light_gray = np.array([0.9, 0.9, 0.9])
    dark_gray = np.array([0.2, 0.2, 0.2])

    # Interpolate between light-gray and dark-gray
    gradient = np.linspace(light_gray, dark_gray, N)

    # Return the gradient as a list of RGB tuples
    return gradient

