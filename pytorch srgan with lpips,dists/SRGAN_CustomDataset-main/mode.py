import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from losses import TVLoss, perceptual_loss
from dataset import *
from srgan_model import Generator, Discriminator
from vgg19 import vgg19
import numpy as np
from PIL import Image
from skimage.color import rgb2ycbcr
#from skimage.measure import compare_psnr
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

import pandas as pd
from skimage.metrics import structural_similarity as compare_ssim
import lpips
import torch
import numpy as np
from DISTS_pytorch import DISTS_pt


def calculate_metrics(gt, output, device, scale=1):
    
    # Convert the images to the range [0, 1]
    output = (output + 1.0) / 2.0
    gt = (gt + 1.0) / 2.0

    # Detach tensors from the computation graph before converting to numpy
    output = output[0].detach().cpu().numpy().transpose(1, 2, 0)  # Shape: (H, W, C)
    gt = gt[0].detach().cpu().numpy().transpose(1, 2, 0)  # Shape: (H, W, C)

    # Calculate PSNR
    y_output = rgb2ycbcr(output)[scale:-scale, scale:-scale, :1]
    y_gt = rgb2ycbcr(gt)[scale:-scale, scale:-scale, :1]
    psnr = compare_psnr(y_output / 255.0, y_gt / 255.0, data_range=1.0)

    # Calculate SSIM with smaller window size if the image size is small
    min_side = min(y_output.shape[0], y_output.shape[1])
    win_size = min(7, min_side)  # Ensure win_size is not larger than the smallest dimension
    if win_size % 2 == 0:  # Ensure win_size is odd
        win_size -= 1

    ssim = compare_ssim(y_output, y_gt, data_range=1.0, channel_axis=-1, win_size=win_size)

    # Calculate LPIPS
    lpips_loss = lpips.LPIPS(net='vgg').to(device)

    # Ensure the tensors have the correct shape (batch_size, 3, H, W)
    output_tensor = torch.tensor(output).permute(2, 0, 1).unsqueeze(0).to(device)  # Shape: (1, C, H, W)
    gt_tensor = torch.tensor(gt).permute(2, 0, 1).unsqueeze(0).to(device)  # Shape: (1, C, H, W)

    # If the tensors have more than 3 channels, select the first 3 channels
    if output_tensor.shape[1] > 3:
        output_tensor = output_tensor[:, :3, :, :]
    if gt_tensor.shape[1] > 3:
        gt_tensor = gt_tensor[:, :3, :, :]

    lpips_score = lpips_loss.forward(output_tensor, gt_tensor).item()

    # Calculate DISTS (using pre-trained DISTS model)
    # Ensure you have installed the DISTS library and import the required model here
    dists_model = DISTS_pt.DISTS().to(device)
    dists_score = dists_model.forward(output_tensor, gt_tensor).item()

    return psnr, ssim, lpips_score, dists_score


def train(args):
    # Initialize CSV file for storing metrics
    metrics_df = pd.DataFrame(columns=["epoch", "psnr", "ssim", "lpips", "dists"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Ensure device is defined
    transform = transforms.Compose([crop(args.scale, args.patch_size), augmentation()])
    dataset = mydata(GT_path=args.GT_path, LR_path=args.LR_path, in_memory=args.in_memory, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    generator = Generator(img_feat=3, n_feats=64, kernel_size=3, num_block=args.res_num, scale=args.scale)
    
    if args.fine_tuning:
        generator.load_state_dict(torch.load(args.generator_path))
        print("pre-trained model is loaded")
        print("path: %s" % (args.generator_path))
    
    generator = generator.to(device)
    generator.train()

    l2_loss = nn.MSELoss()
    g_optim = optim.Adam(generator.parameters(), lr=1e-4)

    pre_epoch = 0
    fine_epoch = 0
    
    # Train using L2_loss
    while pre_epoch < args.pre_train_epoch:
        for i, tr_data in enumerate(loader):
            gt = tr_data['GT'].to(device)
            lr = tr_data['LR'].to(device)

            output, _ = generator(lr)
            loss = l2_loss(gt, output)

            g_optim.zero_grad()
            loss.backward()
            g_optim.step()

        pre_epoch += 1

        if pre_epoch % 2 == 0:
            print(pre_epoch)
            print(loss.item())
            print('=========')

        # Save metrics every epoch
        if pre_epoch % 10 == 0:
            psnr, ssim, lpips_score, dists_score = calculate_metrics(gt, output, device)  # Pass device here
            metrics_df = metrics_df.append({"epoch": pre_epoch, "psnr": psnr, "ssim": ssim, "lpips": lpips_score, "dists": dists_score}, ignore_index=True)
            metrics_df.to_csv('./metrics.csv', index=False)

        if pre_epoch % 100 == 0:
            torch.save(generator.state_dict(), './model/pre_trained_model_%03d.pt' % pre_epoch)

    # Fine-tuning with perceptual & adversarial loss
    vgg_net = vgg19().to(device)
    vgg_net = vgg_net.eval()

    discriminator = Discriminator(patch_size=args.patch_size * args.scale)
    discriminator = discriminator.to(device)
    discriminator.train()

    d_optim = optim.Adam(discriminator.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(g_optim, step_size=2000, gamma=0.1)

    VGG_loss = perceptual_loss(vgg_net)
    cross_ent = nn.BCELoss()
    tv_loss = TVLoss()
    real_label = torch.ones((args.batch_size, 1)).to(device)
    fake_label = torch.zeros((args.batch_size, 1)).to(device)

    while fine_epoch < args.fine_train_epoch:
        scheduler.step()

        for i, tr_data in enumerate(loader):
            gt = tr_data['GT'].to(device)
            lr = tr_data['LR'].to(device)

            # Training Discriminator
            output, _ = generator(lr)
            fake_prob = discriminator(output)
            real_prob = discriminator(gt)

            d_loss_real = cross_ent(real_prob, real_label)
            d_loss_fake = cross_ent(fake_prob, fake_label)

            d_loss = d_loss_real + d_loss_fake

            g_optim.zero_grad()
            d_optim.zero_grad()
            d_loss.backward()
            d_optim.step()

            # Training Generator
            output, _ = generator(lr)
            fake_prob = discriminator(output)

            _percep_loss, hr_feat, sr_feat = VGG_loss((gt + 1.0) / 2.0, (output + 1.0) / 2.0, layer=args.feat_layer)

            L2_loss = l2_loss(output, gt)
            percep_loss = args.vgg_rescale_coeff * _percep_loss
            adversarial_loss = args.adv_coeff * cross_ent(fake_prob, real_label)
            total_variance_loss = args.tv_loss_coeff * tv_loss(args.vgg_rescale_coeff * (hr_feat - sr_feat) ** 2)

            g_loss = percep_loss + adversarial_loss + total_variance_loss + L2_loss

            g_optim.zero_grad()
            d_optim.zero_grad()
            g_loss.backward()
            g_optim.step()

        fine_epoch += 1

        if fine_epoch % 2 == 0:
            print(fine_epoch)
            print(g_loss.item())
            print(d_loss.item())
            print('=========')

        # Save metrics every epoch
        if fine_epoch % 10 == 0:
            psnr, ssim, lpips_score, dists_score = calculate_metrics(gt, output, device)  # Pass device here
            metrics_df = metrics_df.append({"epoch": fine_epoch, "psnr": psnr, "ssim": ssim, "lpips": lpips_score, "dists": dists_score}, ignore_index=True)
            metrics_df.to_csv('./metrics.csv', index=False)

        if fine_epoch % 500 == 0:
            torch.save(generator.state_dict(), './model/SRGAN_gene_%03d.pt' % fine_epoch)
            torch.save(discriminator.state_dict(), './model/SRGAN_discrim_%03d.pt' % fine_epoch)

# Rest of the functions remain the same...





'''




def train(args):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform  = transforms.Compose([crop(args.scale, args.patch_size), augmentation()])
    dataset = mydata(GT_path = args.GT_path, LR_path = args.LR_path, in_memory = args.in_memory, transform = transform)
    loader = DataLoader(dataset, batch_size = args.batch_size, shuffle = True, num_workers = args.num_workers)
    
    generator = Generator(img_feat = 3, n_feats = 64, kernel_size = 3, num_block = args.res_num, scale=args.scale)
    
    
    if args.fine_tuning:        
        generator.load_state_dict(torch.load(args.generator_path))
        print("pre-trained model is loaded")
        print("path : %s"%(args.generator_path))
        
    generator = generator.to(device)
    generator.train()
    
    l2_loss = nn.MSELoss()
    g_optim = optim.Adam(generator.parameters(), lr = 1e-4)
        
    pre_epoch = 0
    fine_epoch = 0
    
    #### Train using L2_loss
    while pre_epoch < args.pre_train_epoch:
        for i, tr_data in enumerate(loader):
            gt = tr_data['GT'].to(device)
            lr = tr_data['LR'].to(device)

            output, _ = generator(lr)
            loss = l2_loss(gt, output)

            g_optim.zero_grad()
            loss.backward()
            g_optim.step()

        pre_epoch += 1

        if pre_epoch % 2 == 0:
            print(pre_epoch)
            print(loss.item())
            print('=========')
        
        if pre_epoch % 100 ==0:
            torch.save(generator.state_dict(), './model/pre_trained_model_%03d.pt'%pre_epoch)

        
    #### Train using perceptual & adversarial loss
    vgg_net = vgg19().to(device)
    vgg_net = vgg_net.eval()
    
    discriminator = Discriminator(patch_size = args.patch_size * args.scale)
    discriminator = discriminator.to(device)
    discriminator.train()
    
    d_optim = optim.Adam(discriminator.parameters(), lr = 1e-4)
    scheduler = optim.lr_scheduler.StepLR(g_optim, step_size = 2000, gamma = 0.1)
    
    VGG_loss = perceptual_loss(vgg_net)
    cross_ent = nn.BCELoss()
    tv_loss = TVLoss()
    real_label = torch.ones((args.batch_size, 1)).to(device)
    fake_label = torch.zeros((args.batch_size, 1)).to(device)
    
    while fine_epoch < args.fine_train_epoch:
        
        scheduler.step()
        
        for i, tr_data in enumerate(loader):
            gt = tr_data['GT'].to(device)
            lr = tr_data['LR'].to(device)
                        
            ## Training Discriminator
            output, _ = generator(lr)
            fake_prob = discriminator(output)
            real_prob = discriminator(gt)
            
            d_loss_real = cross_ent(real_prob, real_label)
            d_loss_fake = cross_ent(fake_prob, fake_label)
            
            d_loss = d_loss_real + d_loss_fake

            g_optim.zero_grad()
            d_optim.zero_grad()
            d_loss.backward()
            d_optim.step()
            
            ## Training Generator
            output, _ = generator(lr)
            fake_prob = discriminator(output)
            
            _percep_loss, hr_feat, sr_feat = VGG_loss((gt + 1.0) / 2.0, (output + 1.0) / 2.0, layer = args.feat_layer)
            
            L2_loss = l2_loss(output, gt)
            percep_loss = args.vgg_rescale_coeff * _percep_loss
            adversarial_loss = args.adv_coeff * cross_ent(fake_prob, real_label)
            total_variance_loss = args.tv_loss_coeff * tv_loss(args.vgg_rescale_coeff * (hr_feat - sr_feat)**2)
            
            g_loss = percep_loss + adversarial_loss + total_variance_loss + L2_loss
            
            g_optim.zero_grad()
            d_optim.zero_grad()
            g_loss.backward()
            g_optim.step()

            
        fine_epoch += 1

        if fine_epoch % 2 == 0:
            print(fine_epoch)
            print(g_loss.item())
            print(d_loss.item())
            print('=========')

        if fine_epoch % 500 ==0:
            #torch.save(generator.state_dict(), './model/SRGAN_gene_%03d.pt'%fine_epoch)
            #torch.save(discriminator.state_dict(), './model/SRGAN_discrim_%03d.pt'%fine_epoch)
            torch.save(generator.state_dict(), './model/SRGAN_gene_%03d.pt'%fine_epoch)
            torch.save(discriminator.state_dict(), './model/SRGAN_discrim_%03d.pt'%fine_epoch)

'''
# In[ ]:

def test(args):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = mydata(GT_path = args.GT_path, LR_path = args.LR_path, in_memory = False, transform = None)
    loader = DataLoader(dataset, batch_size = 1, shuffle = False, num_workers = args.num_workers)
    
    generator = Generator(img_feat = 3, n_feats = 64, kernel_size = 3, num_block = args.res_num)
    generator.load_state_dict(torch.load(args.generator_path))
    generator = generator.to(device)
    generator.eval()
    
    f = open('./result.txt', 'w')
    psnr_list = []
    
    with torch.no_grad():
        for i, te_data in enumerate(loader):
            gt = te_data['GT'].to(device)
            lr = te_data['LR'].to(device)

            bs, c, h, w = lr.size()
            gt = gt[:, :, : h * args.scale, : w *args.scale]

            output, _ = generator(lr)

            output = output[0].cpu().numpy()
            output = np.clip(output, -1.0, 1.0)
            gt = gt[0].cpu().numpy()

            output = (output + 1.0) / 2.0
            gt = (gt + 1.0) / 2.0

            output = output.transpose(1,2,0)
            gt = gt.transpose(1,2,0)

            y_output = rgb2ycbcr(output)[args.scale:-args.scale, args.scale:-args.scale, :1]
            y_gt = rgb2ycbcr(gt)[args.scale:-args.scale, args.scale:-args.scale, :1]
            
            psnr = compare_psnr(y_output / 255.0, y_gt / 255.0, data_range = 1.0)
            psnr_list.append(psnr)
            f.write('psnr : %04f \n' % psnr)

            result = Image.fromarray((output * 255.0).astype(np.uint8))
            result.save('./result/res_%04d.png'%i)

        f.write('avg psnr : %04f' % np.mean(psnr_list))


def test_only(args):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = testOnly_data(LR_path = args.LR_path, in_memory = False, transform = None)
    loader = DataLoader(dataset, batch_size = 1, shuffle = False, num_workers = args.num_workers)
    
    generator = Generator(img_feat = 3, n_feats = 64, kernel_size = 3, num_block = args.res_num)
    generator.load_state_dict(torch.load(args.generator_path))
    generator = generator.to(device)
    generator.eval()
    
    with torch.no_grad():
        for i, te_data in enumerate(loader):
            lr = te_data['LR'].to(device)
            output, _ = generator(lr)
            output = output[0].cpu().numpy()
            output = (output + 1.0) / 2.0
            output = output.transpose(1,2,0)
            result = Image.fromarray((output * 255.0).astype(np.uint8))
            result.save('./result/res_%04d.png'%i)



