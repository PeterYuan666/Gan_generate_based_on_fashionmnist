import torch
import torch.nn as nn
import time
image_size = [1, 28, 28]
from torchvision.utils import save_image

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Tanh(),
        )
    def forward(self, z):
        output = self.model(z)
        image = output.reshape(z.shape[0], *image_size)
        return image

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
    def forward(self, image):
        posibility = self.model(image.reshape(image.shape[0], -1))
        return posibility

def load_pretrained_generator():
    generator = Generator()
    generator.load_state_dict(torch.load('pretrained_generator.pth', map_location=torch.device('cpu')))
    generator.eval()  
    return generator


def load_pretrained_discriminator():
    discriminator = Discriminator()
    discriminator.load_state_dict(torch.load('pretrained_discriminator.pth', map_location=torch.device('cpu')))
    discriminator.eval() 
    return discriminator

lantern = 784
z = torch.randn(12, lantern)
z_images = z.reshape(z.shape[0], *image_size)
z_images = z_images.view(-1, 1, 28, 28)
save_image(z_images, 'initial_2.png', nrow=12)  # Save image
print('Initial image has been saved')

generator = load_pretrained_generator()
discriminator = load_pretrained_discriminator()
sample_images = generator(z)
time.sleep(6)
l1 = discriminator(z)
print(l1, 'For randomly generated images judgment')
sample_images = sample_images.view(-1, 1, 28, 28)  # Adjust dimensions to match the image
filename = f'generated_images_4.png'  # Construct filename
save_image(sample_images, filename, nrow=12)  # Save image
l2 = discriminator(sample_images)
print(l2, 'For transformed image judgment')
print(f'The image has been saved as {filename}')
