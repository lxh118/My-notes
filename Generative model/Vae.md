# Variational Autoencoder (VAE) Overview

## Core Concept
VAE is a generative model that aims to learn the latent distribution of data and generate new data samples.

- **Autoencoder (AE)**: Directly learns a compressed representation (encoding) of data and reconstructs it through a decoder, without explicitly modeling the latent probability distribution.
- **Variational Autoencoder (VAE)**: Maps input data to a latent probability distribution (typically Gaussian) and generates new data by sampling.

## Theoretical Framework

### (1) Probabilistic Generative Model
VAE assumes data is generated through:
1. Sampling a latent variable \( z \) from prior distribution \( p(z) \)
2. Generating data \( x \) through decoder \( p_{\theta}(x|z) \)

The goal is to maximize the marginal likelihood \( p_{\theta}(x) \), but direct computation is intractable (requires integration over all possible \( z \)).

### (2) Variational Inference
VAE introduces an approximate posterior distribution \( q_{\phi}(z|x) \) (encoder) to solve the computational challenge. The optimization objective is the Evidence Lower Bound (ELBO):

\[
\log p_{\theta}(x) \geq \text{ELBO} = \mathbb{E}_{q_{\phi}(z|x)}[\log p_{\theta}(x|z)] - D_{\text{KL}}(q_{\phi}(z|x) \| p(z))
\]

- **First term**: Reconstruction loss (similarity between generated and original data)
- **Second term**: KL divergence, constraining \( q_{\phi}(z|x) \) to be close to prior \( p(z) \)

## Model Architecture
VAE consists of two components:

### Encoder
- Input: data \( x \), outputs distribution parameters (mean and variance) for latent variable \( z \)
- Example: \( q_{\phi}(z|x) = \mathcal{N}(\mu_{\phi}(x), \sigma_{\phi}(x)) \)
- Implementation: Typically uses neural networks to predict \( \mu \) and \( \log \sigma^2 \)

### Decoder
- Samples \( z \) from latent space and generates data distribution \( x \)
- Example: \( p_{\theta}(x|z) \) can be Bernoulli (binary data) or Gaussian (continuous data)

## Reparameterization Trick
Enables gradient backpropagation through sampling:

\[
z = \mu_{\phi}(x) + \sigma_{\phi}(x) \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
\]

The randomness comes from external variable \( \epsilon \), allowing optimization of \( \mu \) and \( \sigma \).

## Training Process

1. **Forward Propagation:**
   - Input data \( x \) → encoder outputs \( \mu, \sigma \) → sample \( z \) → decoder generates \( x' \)

2. **Loss Function:**
   \[
   \mathcal{L}(\theta, \phi) = -\text{ELBO} = \text{Reconstruction Loss} + \text{KL Divergence}
   \]
   - Reconstruction loss: e.g., binary cross-entropy (image generation) or mean squared error
   - KL divergence: Closed-form solution \( -\frac{1}{2} \sum (1 + \log \sigma^2 - \mu^2 - \sigma^2) \)

3. **Backpropagation:** Gradients computed via reparameterization trick, updating encoder and decoder parameters.

## Code Implementation

```python
class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)  # Output μ and logσ²
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  # For [0,1] input data
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encoding
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=-1)
        z = self.reparameterize(mu, logvar)
        # Decoding
        x_recon = self.decoder(z)
        return x_recon, mu, logvar

# Loss Function
def loss_function(x_recon, x, mu, logvar):
    recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss
