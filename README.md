
# GAN Generated Images Based on Fashion-MNIST  
  
This section showcases the progress of GAN (Generative Adversarial Network) training, from initial noise to generated images resembling the Fashion-MNIST dataset.  
  
## Initial Noise (or Starting Point)  
  
![Initial Noise](https://github.com/PeterYuan666/Gan_generate_based_on_fashionmnist/blob/cbd006c1cfa36278148f04535924662769cf170c/initial.png)  
  
* This is typically where the GAN starts, with random noise as input.  
  
## Intermediate Generation  
  
![Intermediate Generation](https://github.com/PeterYuan666/Gan_generate_based_on_fashionmnist/blob/cbd006c1cfa36278148f04535924662769cf170c/middle_generated_images.png)  
  
* As the GAN trains, the generated images start to take shape and become more recognizable.  
  
## Final Generated Images  
  
![Final Generated Images](https://github.com/PeterYuan666/Gan_generate_based_on_fashionmnist/blob/cbd006c1cfa36278148f04535924662769cf170c/final_generated_images.png)  
  
* After sufficient training, the GAN is able to produce images that resemble the Fashion-MNIST dataset.  
  
## Description  
  
The GAN was trained for [800] epochs on the Fashion-MNIST dataset, gradually improving its ability to generate realistic images of clothing items. The above images represent key stages in this process, from the initial random noise to the final generated outputs.

# Project Overview

This project extends the Generative Adversarial Network (GAN) model built on PyTorch, utilizing pre-trained Generator and Discriminator to generate images resembling those found in the FashionMNIST dataset.

## Environment Setup

Ensure your environment has the following Python libraries installed:

- PyTorch
- torchvision

You can install these libraries (if not already installed) using pip:

```bash
pip install torch torchvision
```

## Code Structure

- `generator.py`: Contains the Generator class responsible for generating images.
- `discriminator.py`: Contains the Discriminator class responsible for distinguishing between real and fake images.
- `train.py` (hypothetical): Contains the code for training the GAN, resulting in pre-trained models.

The README also includes relevant scripts or code snippets for loading pre-trained models and generating images.

## Key Components

### Generator

- **Input**: A random noise vector (shape: `[batch_size, 784]`)
- **Output**: Generated images (shape: `[batch_size, 1, 28, 28]`)

### Discriminator

- **Input**: Images (shape: `[batch_size, 1, 28, 28]`)
- **Output**: Probability that the image is real (shape: `[batch_size, 1]`)

## Generating Images Using Pre-trained Models

The following steps outline how to use the pre-trained Generator (and optionally the Discriminator, though primarily the Generator is used for image generation) to generate images:

1. **Load Pre-trained Models**: Use `load_pretrained_generator` and `load_pretrained_discriminator` (optional) functions to load the models.
2. **Generate Random Noise**: Create a random noise vector as input for the Generator.
3. **Generate Images**: Feed the random noise into the Generator to obtain generated images.
4. **Save Images**: Use `torchvision.utils.save_image` to save the generated images.

### Example Code

```python
# Assuming load_pretrained_generator and load_pretrained_discriminator are defined in appropriate modules
from some_module import load_pretrained_generator, load_pretrained_discriminator  # Replace with correct module names
from torchvision.utils import save_image
import torch

# Load pre-trained models
generator = load_pretrained_generator()
# The Discriminator is usually not needed for image generation but shown for completeness
discriminator = load_pretrained_discriminator()  # Optional

# Generate random noise
latent_dim = 784  # Dimension of input noise
z = torch.randn(12, latent_dim)  # Assuming we want to generate 12 images

# Generate images
with torch.no_grad():  # Ensure not in training mode
    generated_images = generator(z)

# Save the generated images
save_image(generated_images, 'generated_images.png', nrow=12)
print('Generated images have been saved.')
```

## Notes

- Ensure the pre-trained model files (`pretrained_generator.pth` and `pretrained_discriminator.pth`) exist in the same directory as your code or at the specified paths.
- The quality of the generated images depends on the performance of the pre-trained models.
- For further exploration or adjustments to the models, you may need to review the original `train.py` script or related configurations.

## Conclusion

By leveraging pre-trained Generative Adversarial Network models, we can quickly generate images resembling those in the FashionMNIST dataset. This technique has broad applications in image generation, data augmentation, artistic creation, and more.
