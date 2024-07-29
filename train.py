# GAN model implementation based on FashionMnist
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image
from tqdm import tqdm
torch.cuda.empty_cache()
image_size = [1, 28, 28]

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
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
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
    def forward(self, image):
        posibility = self.model(image.reshape(image.shape[0], -1))
        return posibility

Dataset = torchvision.datasets.FashionMNIST(
    'fashionmnist_data', train=True, download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
)
dataloader = DataLoader(Dataset, batch_size=32, shuffle=True)

# Assuming Generator and Discriminator classes are already defined
generator = Generator()
discriminator = Discriminator()

def train(batch_size, num_epoch, lr):
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=lr)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr)
    target1 = torch.ones(batch_size, 1)
    target2 = torch.zeros(batch_size, 1)
    lantern_dim = 784
    loss_function = nn.BCELoss()

    if torch.cuda.is_available():
        target1, target2 = target1.cuda(), target2.cuda()
        generator.cuda()
        discriminator.cuda()

    for epoch in tqdm(range(num_epoch), desc="Epoch"):
        for i, minibatch in enumerate(dataloader):
            gt_images, _ = minibatch

            if torch.cuda.is_available():
                gt_images = gt_images.cuda()

            z = torch.randn(batch_size, lantern_dim)
            if torch.cuda.is_available():
                z = z.cuda()

            pred_images = generator(z)
            g_optimizer.zero_grad()
            g_loss = loss_function(discriminator(pred_images), target1)
            g_loss.backward()
            g_optimizer.step()

            # Optimizing the discriminator below
            d_optimizer.zero_grad()
            real_loss = loss_function(discriminator(gt_images), target1)
            fake_loss = loss_function(discriminator(pred_images.detach()), target2)
            d_loss = real_loss + fake_loss
            d_loss.backward()
            d_optimizer.step()

        if (epoch) % 50 == 0:
                print('Generator loss is g_loss=', g_loss)
                print('Discriminator loss is d_loss', d_loss)
                with torch.no_grad():
                    a = torch.randn(10, lantern_dim)
                    a = a.cuda()
                    sample_images = generator(a)
                    if torch.cuda.is_available():
                        sample_images = sample_images.cpu()
                    sample_images = sample_images.view(-1, 1, 28, 28)
                    epoch_index = (epoch) // 50
                    filename = f'generated_images_epoch_{epoch_index}.png'
                    save_image(sample_images, filename, nrow=10)
                    print(f'Images from epoch {epoch_index} saved as {filename}')
        if g_loss < 0.1 and d_loss < 0.05:
                break
    torch.save(generator.state_dict(), 'pretrained_generator.pth')
    torch.save(discriminator.state_dict(), 'pretrained_discriminator.pth')
train(32,800,0.01)
def load_pretrained_generator():
    generator = Generator()
    generator.load_state_dict(torch.load('pretrained_generator.pth'))
    generator.eval()  # Set model to evaluation mode
    return generator

def load_pretrained_discriminator():
    discriminator = Discriminator()
    discriminator.load_state_dict(torch.load('pretrained_discriminator.pth'))
    discriminator.eval()  # Set model to evaluation mode
    return discriminator

