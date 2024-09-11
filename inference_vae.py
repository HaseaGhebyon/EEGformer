import torch
from torch.distributions.normal import Normal
from model_vae import VAE
import numpy as np
import matplotlib.pyplot as plt
def plot_latent_images(model, n, digit_size=28, z_dim=32):
    norm = Normal(0,1)
    grid_x = norm.icdf(torch.linspace(0.005, 0.995, n))
    grid_y = norm.icdf(torch.linspace(0.005, 0.995, n))
    image_width = digit_size*n
    image_height = image_width
    image = np.zeros((image_height, image_width))

    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z = torch.tensor([[xi] * (z_dim // 2) + [yi] * (z_dim // 2)])  # Membuat tensor untuk latent space z
            z = z.to(next(model.parameters()).device)  # Memastikan tensor ada di device yang sama dengan model
            z = z.view(1, z_dim, 1, 1)  # Mengatur dimensi tensor z
            x_decoded = model.decode(z)  # Mendekodekan z ke gambar
            digit = x_decoded.view(digit_size, digit_size).detach().cpu().numpy()  # Mengubah tensor menjadi numpy array
            image[i * digit_size: (i + 1) * digit_size,
                  j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(20, 20))
    plt.imshow(image, cmap='Greys_r')
    plt.axis('Off')
    plt.savefig('vae_plot.png', bbox_inches='tight')

model = VAE()
model = model.to('cuda:0')
snapshot = torch.load("./database_vae/vae_weights/vaemodel_100")
model.load_state_dict(snapshot["MODEL_STATE"])
plot_latent_images(model, n=20)