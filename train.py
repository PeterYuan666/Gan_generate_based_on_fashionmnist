#基于FashionMnist实现GAN模型的代码实现
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image
from tqdm import tqdm
torch.cuda.empty_cache()
image_size=[1,28,28]
class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.model=nn.Sequential(
            nn.Linear(784,64),
            nn.ReLU(),
            nn.Linear(64, 256),
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
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32,1),
            nn.Sigmoid(),
        )
    def forward(self,image):
        #image=[batch_size,1,28,28],输入的图片（数据集加生成）
        posibility=self.model(image.reshape(image.shape[0],-1))
        return posibility
Dataset = torchvision.datasets.FashionMNIST(  
    'fashionmnist_data', train=True, download=True,  
    transform=transforms.Compose([  
        transforms.ToTensor(),  
        transforms.Normalize(mean=[0.5], std=[0.5])  
    ])  
)
dataloader = DataLoader(Dataset, batch_size=32, shuffle=True)

        # 假设Generator和Discriminator类已经定义好
generator = Generator()
discriminator = Discriminator()

def train(batch_size,num_epoch, lr):
            g_optimizer = torch.optim.Adam(generator.parameters(), lr=lr)
            d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr)
            target1 = torch.ones(batch_size, 1)  # 假设batch_size为32
            target2 = torch.zeros(batch_size, 1)
            lantern_dim = 784
            loss_function = nn.BCELoss()

            if torch.cuda.is_available():
                target1, target2 = target1.cuda(), target2.cuda()
                generator.cuda()
                discriminator.cuda()

            for epoch in tqdm(range(num_epoch),desc="Epoch"):
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
                    # 梯度下降+优化
                    g_optimizer.step()
                    # 下面进行判别器优化
                    d_optimizer.zero_grad()
                    real_loss = loss_function(discriminator(gt_images), target1)
                    fake_loss = loss_function(discriminator(pred_images.detach()), target2)  # detach() 防止对生成器进行梯度传播
                    d_loss = real_loss + fake_loss
                    d_loss.backward()
                    d_optimizer.step()

                if (epoch) % 50 == 0:
                        print('生成器中损失为g_loss=', g_loss)
                        print('判别器中损失为d_loss', d_loss)
                        with torch.no_grad():
                            a=torch.randn(10, lantern_dim)
                            a=a.cuda()
                            sample_images = generator(a)
                            if torch.cuda.is_available():
                                sample_images = sample_images.cpu()
                            sample_images = sample_images.view(-1, 1, 28, 28)  # 调整维度以匹配图像
                            epoch_index = (epoch) // 50  # 获取轮次索引（每10个epoch为一次）
                            filename = f'generated_images_epoch_{epoch_index}.png'  # 构造文件名
                            save_image(sample_images, filename, nrow=10)  # 保存图像
                            print(f'第{epoch_index}次的图片已保存为 {filename}')
                if g_loss<0.1 and d_loss<0.05:
                    break
            torch.save(generator.state_dict(), 'pretrained_generator.pth')
            torch.save(discriminator.state_dict(), 'pretrained_discriminator.pth')
train(32,1001,0.0002)
def load_pretrained_generator():
        generator = Generator()
        generator.load_state_dict(torch.load('pretrained_generator.pth'))
        generator.eval()  # 设置模型为评估模式
        return generator


def load_pretrained_discriminator():
        discriminator = Discriminator()
        discriminator.load_state_dict(torch.load('pretrained_discriminator.pth'))
        discriminator.eval()  # 设置模型为评估模式
        return discriminator