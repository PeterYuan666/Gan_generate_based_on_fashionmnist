import torch
import torch.nn as nn
import time
image_size=[1,28,28]
from torchvision.utils import save_image
class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.model=nn.Sequential(
            nn.Linear(784,64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512,784),
            nn.Tanh(),
        )
    def forward(self,z):
        #z=[batch_size,1*28*28],相当于预生成的高斯噪声
        #batch_size为一个样本中的样本数
        output=self.model(z)
        image=output.reshape(z.shape[0],*image_size)
        return image
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.model=nn.Sequential(
            nn.Linear(784,512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32,1),
            nn.Sigmoid(),
        )
    def forward(self,image):
        #image=[batch_size,1,28,28],输入的图片（数据集加生成）
        posibility=self.model(image.reshape(image.shape[0],-1))
        return posibility
def load_pretrained_generator():
    generator = Generator()
    generator.load_state_dict(torch.load('pretrained_generator.pth'))
    generator.eval()  # 设置模型为生成模式
    return generator


def load_pretrained_discriminator():
    discriminator = Discriminator()
    discriminator.load_state_dict(torch.load('pretrained_discriminator.pth'))
    discriminator.eval()  # 设置模型为评估模式
    return discriminator
lantern=784
z = torch.randn(12,lantern)
z_images=z.reshape(z.shape[0],*image_size)
z_images=z_images.view(-1,1,28,28)
save_image(z_images, 'initial_2.png', nrow=12)  # 保存图像
print('initial_image已经保存')
generator=load_pretrained_generator()
discriminator=load_pretrained_discriminator()
sample_images=generator(z)
time.sleep(6)
l1=discriminator(z)
print(l1,'对于随机生成的图的判断')
sample_images = sample_images.view(-1, 1, 28, 28)  # 调整维度以匹配图像
filename = f'generated_images_4.png'  # 构造文件名
save_image(sample_images, filename, nrow=12)  # 保存图像
l2=discriminator(sample_images)
print(l2,'对于转化之后图的判断')
print(f'第图片已保存为 {filename}')
