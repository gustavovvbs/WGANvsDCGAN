# WGANvsDCGAN

DCGANs (Deep Convolutional Adversarial Networks) are a widely studied architeture in the field of generative adversarial networks. They are known for their ability to generate realistic synthetic data. However, one of the main challenges with DCGANs is their instability during training, often resulting in issues such as mode collapse, where the generator produces limited varities of outputs, or the inability of the generator and discriminator to reach a state equilibrium.

To mitigate these challenges, the Wasserstein GAN (WGAN) was introduced, and later improved with techniques such as gradient penalty and spectral normalization of the weight tensors. These changes aims to provide more stability during the training process of the network.

## DCGAN Architeture

The architeture of a DCGAN is typically composed of a generator and a discriminator, with the generator using de-convolutional layers, and the discriminator using convolutional layers. The generator maps a random noise vector sampled from a latent gaussian distribution to a synthetic image, while the discriminator is trained to classify whether the generated image is real or not.

This training involves a minmax game between the generator and the discriminator, formulated as:

$\[ \min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{data}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))] \]$

However, the loss function in this formulation does not correlate with the quality of generated samples, while their objective is to classify the images as real or generated.

## WGAN

The WGAN architeture introduces the Wasserstein distance (also known as Earth Mover's distance) as a measure of distance between the real data distribution $\(p_{data}\)$ and the generated data distribution $\(p_g\)$. This distance is defined as:

$\[ W(p_{data}, p_g) = \inf_{\gamma \in \Pi(p_{data}, p_g)} \mathbb{E}_{(x, y) \sim \gamma} [\|x - y\|] \]$


Where $\( \Pi(p_{data}, p_g) \)$ denotes the set of all joint distributions whose marginals are $\( p_{data} \)$ and $\( p_g \)$. 
Since the Wasserstein distance is a direct measure of the distance between the distribution of real and fake data, the loss will be able to also judge the quality of generated samples by measuring their distributions compared to the real ones, giving a more meaningful loss function.

## Gradient Penalty 

The wasserstein distance needs a 1-Lipschitz continuity condition in the compared distributions. To ensure this condition, we can use gradient penalty. The goal of the gradient penalty is to regularize the norm of the gradients of the discriminator with respect to its inputs, pushing it towards 1.

### Detailed Computation of the Gradient Penalty

To compute the norm used in the gradient penalty, a random tensor $\( \epsilon \)$ is sampled from a uniform distribution in the range $\( ]0, 1[ \)$. This tensor $\( \epsilon \)$ serves as a mixing coefficient to interpolate between real and generated samples.

For each real data sample $\( x_{real} \)$ and generated data sample $\( x_{fake} \)$, we compute the interpolated sample $\( \hat{x} \)$ as:

$\[ \hat{x} = \epsilon \cdot x_{real} + (1 - \epsilon) \cdot x_{fake} \]$

This interpolation ensures that $\( \hat{x} \)$ lies on the line connecting $\( x_{real} \)$ and $\( x_{fake} \)$ in the input space. 

Next, the interpolated sample $\( \hat{x} \)$ is passed through the discriminator $\( D(\hat{x}) \)$, and the gradient of the discriminator's output with respect to $\( \hat{x} \)$ is computed:

$\[ \nabla_{\hat{x}} D(\hat{x}) \]$

This gradient represents how sensitive the discriminator's output is to changes in $\( \hat{x} \)$. The goal of the gradient penalty is to enforce that the norm of this gradient is close to 1, which is a requirement for the discriminator to be 1-Lipschitz continuous, as needed for the Wasserstein distance calculation.

The gradient norm is computed using the $\( L_2 \)$ norm (also known as the Euclidean norm):

$\[ \|\nabla_{\hat{x}} D(\hat{x})\|_2 \]$

The difference between this norm and 1 is calculated, squared, and then multiplied by the regularization coefficient $\( \lambda \)$. The final gradient penalty term added to the loss function is:

$\[ \lambda \mathbb{E}_{\hat{x} \sim p_{\hat{x}}} [(\|\nabla_{\hat{x}} D(\hat{x})\|_2 - 1)^2] \]$

This penalty encourages the gradient norm to stay close to 1, which helps to stabilize the training process by preventing issues like gradient explosion or vanishing gradients. 

## Spectral Normalization

Spectral normalization is another technique used to ensure the Lipschitz continuity of the discriminator. It involves normalizing the weights of each layer in the discriminator by their largest singular value. Mathematically, this can be represented as:

$\[ \hat{W} = \frac{W}{\sigma(W)} \]$

where $\( W \)$ is the weight matrix, and $\( \sigma(W) \)$ is its largest singular value. This normalization limits the maximum gradient value that can be backpropagated, further stabilizing the training process.
