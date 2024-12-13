import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class RandomResize(nn.Module):
    def __call__(self, x):
        r = random.random()
        resizer = transforms.Resize(int(64 * (1.0 + r*0.15)), Image.BICUBIC, antialias=True)
        return resizer(x)


class Transfroms(torch.nn.Module):
    def __init__(self, shape):
        super(Transfroms, self).__init__()
        assert len(shape) == 3
        transforms_ = [
            RandomResize(),
            transforms.RandomCrop((shape[1], shape[2])),
        ]
        self.transforms = torch.nn.Sequential(*transforms_)
        # Image transformations
        
    def forward(self, x):
        x = self.transforms(x)
        # max_, min_ = torch.amax(x), torch.amin(x)
        # x = (x - min_) / (max_ - min_ + 1e-7)
        # return self.normalizer(x)
        return x
    

class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        channels, height, width = input_shape
        # Calculate output of image discriminator (PatchGAN)
        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters)) # type: ignore
            layers.append(nn.LeakyReLU(0.2, inplace=True)) # type: ignore
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.Conv2d(512, 1, 3, padding=1)
        )

    def forward(self, img):
        return self.model(img)

class CycleVAE(nn.Module):
    def __init__(self, input_size, latent_size):
        super().__init__()

        self.transformer = Transfroms((3, 64, 64))

        self.encoder_1 = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, latent_size),
        )
        # self.fc_mu_1 = nn.Linear(512, latent_size)
        # self.fc_logvar_1 = nn.Linear(512, latent_size)
        self.decoder_1 = nn.Sequential(
            nn.Linear(latent_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, input_size)
        )

        self.encoder_2 = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, latent_size),
        )
        # self.fc_mu_2 = nn.Linear(512, latent_size)
        # self.fc_logvar_2 = nn.Linear(512, latent_size)
        self.decoder_2 = nn.Sequential(
            nn.Linear(latent_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, input_size)
        )

        self.D1 = Discriminator((3, 64, 64))
        self.D2 = Discriminator((3, 64, 64))

        self.apply(weights_init_normal)

        self.optimizer_G = torch.optim.Adam(
            list(self.encoder_1.parameters()) + 
            list(self.encoder_2.parameters()) + 
            list(self.decoder_1.parameters()) + 
            list(self.decoder_2.parameters()), # type: ignore
            lr=0.0001,
            betas=(0.5, 0.999),
        )
        self.optimizer_D1 = torch.optim.Adam(self.D1.parameters(), lr=0.0001, betas=(0.5, 0.999))
        self.optimizer_D2 = torch.optim.Adam(self.D2.parameters(), lr=0.0001, betas=(0.5, 0.999))

        # Losses
        self.criterion_GAN = torch.nn.MSELoss()
        self.criterion_pixel = torch.nn.L1Loss()


    def reparameterize(self, mu):
        std = torch.randn_like(mu)
        return mu + std


    def forward(self, args, agent, agent_2, a_1, a_2, observations, observations_2):
        x_1 = agent.encoder(observations)
        mu_1 = self.encoder_1(x_1)
        z_1 = self.reparameterize(mu_1)
        recon_1 = self.decoder_1(z_1)
        belief, _, _, _, posterior_state, _, _ = agent.transition_model(
          torch.zeros(1, args.state_size, device=args.device),
          a_1,
          torch.zeros(1, args.belief_size, device=args.device),
          recon_1.unsqueeze(dim=0))
        belief, posterior_state = belief.squeeze(dim=0), posterior_state.squeeze(dim=0)
        reconst_observation = agent.observation_model(belief, posterior_state)

        x_2 = agent_2.encoder(observations_2)
        mu_2 = self.encoder_2(x_2)
        z_2 = self.reparameterize(mu_2)
        recon_2 = self.decoder_2(z_2)
        belief, _, _, _, posterior_state, _, _ = agent_2.transition_model(
            torch.zeros(1, args.state_size, device=args.device),
            a_2,
            torch.zeros(1, args.belief_size, device=args.device),
            recon_2.unsqueeze(dim=0))
        belief, posterior_state = belief.squeeze(dim=0), posterior_state.squeeze(dim=0)
        reconst_observation_2 = agent_2.observation_model(belief, posterior_state)

        recon_2_from_1_ = self.decoder_2(z_1)
        belief, _, _, _, posterior_state, _, _ = agent_2.transition_model(
            torch.zeros(1, args.state_size, device=args.device),
            a_1,
            torch.zeros(1, args.belief_size, device=args.device),
            recon_2_from_1_.unsqueeze(dim=0))
        belief, posterior_state = belief.squeeze(dim=0), posterior_state.squeeze(dim=0)
        reconst_2_from_1 = agent_2.observation_model(belief, posterior_state)
        recon_2_from_1_ = agent_2.encoder(reconst_2_from_1)
        mu_2_ = self.encoder_2(recon_2_from_1_)
        z_2_ = self.reparameterize(mu_2_)
        recon_2_from_1 = self.decoder_1(z_2_)
        belief, _, _, _, posterior_state, _, _ = agent.transition_model(
            torch.zeros(1, args.state_size, device=args.device),
            a_1,
            torch.zeros(1, args.belief_size, device=args.device),
            recon_2_from_1.unsqueeze(dim=0))
        belief, posterior_state = belief.squeeze(dim=0), posterior_state.squeeze(dim=0)
        reconst_observation_1_from_2 = agent.observation_model(belief, posterior_state)

        recon_1_from_2_ = self.decoder_1(z_2)
        belief, _, _, _, posterior_state, _, _ = agent.transition_model(
            torch.zeros(1, args.state_size, device=args.device),
            a_2,
            torch.zeros(1, args.belief_size, device=args.device),
            recon_1_from_2_.unsqueeze(dim=0))
        belief, posterior_state = belief.squeeze(dim=0), posterior_state.squeeze(dim=0)
        reconst_1_from_2 = agent.observation_model(belief, posterior_state)
        recon_1_from_2_ = agent.encoder(reconst_1_from_2)
        mu_1_ = self.encoder_1(recon_1_from_2_)
        z_1_ = self.reparameterize(mu_1_)
        recon_1_from_2 = self.decoder_2(z_1_)
        belief, _, _, _, posterior_state, _, _ = agent_2.transition_model(
            torch.zeros(1, args.state_size, device=args.device),
            a_2,
            torch.zeros(1, args.belief_size, device=args.device),
            recon_1_from_2.unsqueeze(dim=0))
        belief, posterior_state = belief.squeeze(dim=0), posterior_state.squeeze(dim=0)
        reconst_observation_2_from_1 = agent_2.observation_model(belief, posterior_state)
        
        return reconst_observation, reconst_observation_2, reconst_observation_1_from_2, reconst_observation_2_from_1, mu_1, mu_2, mu_1_, mu_2_, reconst_2_from_1, reconst_1_from_2
    
    @staticmethod
    def compute_kl(mu):
        mu_2 = torch.pow(mu, 2)
        loss = torch.mean(mu_2)
        return loss

    @staticmethod
    def reconstruction_loss(recon_x, x):
        return F.mse_loss(
                recon_x,
                x,
                reduction='none').sum((1,2,3)).mean()
    
    @staticmethod
    def kl_divergence(mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    @staticmethod
    def cycle_consistency_loss(x, recon_x_from_y, y, recon_y_from_x):
        loss_X_to_Y_to_X = CycleVAE.reconstruction_loss(recon_x_from_y, x)
        loss_Y_to_X_to_Y = CycleVAE.reconstruction_loss(recon_y_from_x, y)
        return loss_X_to_Y_to_X + loss_Y_to_X_to_Y


    def update_parameters(self, epochs, agent, agent_2, D, D_2, args, episode):
        batch_size = 1

        # Loss weights
        lambda_0 = 10  # GAN
        lambda_1 = 0.1  # KL (encoded images)
        lambda_2 = 100  # ID pixel-wise
        lambda_3 = 0.1  # KL (encoded translated images)
        lambda_4 = 100  # Cycle pixel-wise

        # Adversarial ground truths
        valid = torch.ones((batch_size, *self.D1.output_shape), requires_grad=False, device=args.device)
        fake = torch.zeros((batch_size, *self.D1.output_shape), requires_grad=False, device=args.device)

        total_loss = 0
        # Freeze the first and third models
        for param in agent.transition_model.parameters():
            param.requires_grad = False
        for param in agent.observation_model.parameters():
            param.requires_grad = False
        for param in agent.encoder.parameters():
            param.requires_grad = False
        
        for param in agent_2.transition_model.parameters():
            param.requires_grad = False
        for param in agent_2.observation_model.parameters():
            param.requires_grad = False
        for param in agent_2.encoder.parameters():
            param.requires_grad = False

        self.train()
        for i in tqdm(range(epochs)):
            observations, a_1, _, _ = D.sample(batch_size, 1)
            observations_2, a_2, _, _ = D_2.sample(batch_size, 1)

            obses = torch.cat([observations[0], observations_2[0]])
            obses = self.transformer(obses)
            observations = obses[[0]]
            observations_2 = obses[[1]]

            self.optimizer_G.zero_grad()
        
            reconst_observation, reconst_observation_2, \
            reconst_observation_1_from_2, reconst_observation_2_from_1, \
            mu_1, mu_2, mu_1_, mu_2_, \
            reconst_2_from_1, reconst_1_from_2= self.forward(args, agent, agent_2, a_1, a_2, observations, observations_2)

            loss_GAN_1 = lambda_0 * self.criterion_GAN(self.D1(reconst_1_from_2), valid)
            loss_GAN_2 = lambda_0 * self.criterion_GAN(self.D2(reconst_2_from_1), valid)
            loss_KL_1 = lambda_1 * self.compute_kl(mu_1)
            loss_KL_2 = lambda_1 * self.compute_kl(mu_2)
            loss_ID_1 = lambda_2 * self.criterion_pixel(observations[0], reconst_observation[0])
            loss_ID_2 = lambda_2 * self.criterion_pixel(observations_2[0], reconst_observation_2[0])
            loss_KL_1_ = lambda_3 * self.compute_kl(mu_1_)
            loss_KL_2_ = lambda_3 * self.compute_kl(mu_2_)
            loss_cyc_1 = lambda_4 * self.criterion_pixel(observations[0], reconst_observation_1_from_2[0])
            loss_cyc_2 = lambda_4 * self.criterion_pixel(observations_2[0], reconst_observation_2_from_1[0])

            # Total loss
            loss_G = (
                loss_KL_1
                + loss_KL_2
                + loss_ID_1
                + loss_ID_2
                + loss_GAN_1
                + loss_GAN_2
                + loss_KL_1_
                + loss_KL_2_
                + loss_cyc_1
                + loss_cyc_2
            )

            loss_G.backward()
            self.optimizer_G.step()
            
            self.optimizer_D1.zero_grad()
            loss_D1 = self.criterion_GAN(self.D1(observations), valid) + self.criterion_GAN(self.D1(reconst_1_from_2.detach()), fake)
            loss_D1.backward()
            self.optimizer_D1.step()

            # -----------------------
            #  Train Discriminator 2
            # -----------------------

            self.optimizer_D2.zero_grad()
            loss_D2 = self.criterion_GAN(self.D2(observations_2), valid) + self.criterion_GAN(self.D2(reconst_2_from_1.detach()), fake)
            loss_D2.backward()
            self.optimizer_D2.step()
            
            total_loss += np.array([
                loss_GAN_1.item(), loss_GAN_2.item(),
                loss_KL_1.item(), loss_KL_2.item(), 
                loss_ID_1.item(), loss_ID_2.item(),
                loss_KL_1_.item(), loss_KL_2_.item(), 
                loss_cyc_1.item(), loss_cyc_2.item(),
                loss_D1.item(), loss_D2.item()])

            if i % 100 == 99:
                fig, ax = plt.subplots(4, 3, figsize=(3, 3))
                ax[0,0].imshow(observations[0].permute(1, 2, 0).detach().cpu().numpy()+0.5)
                ax[0,1].imshow(reconst_observation[0].permute(1, 2, 0).detach().cpu().numpy()+0.5)
                
                ax[1,0].imshow(observations[0].permute(1, 2, 0).detach().cpu().numpy()+0.5)
                ax[1,1].imshow(reconst_2_from_1[0].permute(1, 2, 0).detach().cpu().numpy()+0.5)
                ax[1,2].imshow(reconst_observation_1_from_2[0].permute(1, 2, 0).detach().cpu().numpy()+0.5)

                ax[2,0].imshow(observations_2[0].permute(1, 2, 0).detach().cpu().numpy()+0.5)
                ax[2,1].imshow(reconst_observation_2[0].permute(1, 2, 0).detach().cpu().numpy()+0.5)

                ax[3,0].imshow(observations_2[0].permute(1, 2, 0).detach().cpu().numpy()+0.5)
                ax[3,1].imshow(reconst_1_from_2[0].permute(1, 2, 0).detach().cpu().numpy()+0.5)
                ax[3,2].imshow(reconst_observation_2_from_1[0].permute(1, 2, 0).detach().cpu().numpy()+0.5)
                fig.savefig('./results/RRR' + str(episode) + ".png")
                plt.close()

        for param in agent.transition_model.parameters():
            param.requires_grad = True
        for param in agent.observation_model.parameters():
            param.requires_grad = True
        for param in agent.encoder.parameters():
            param.requires_grad = True

        for param in agent_2.transition_model.parameters():
            param.requires_grad = True
        for param in agent_2.observation_model.parameters():
            param.requires_grad = True
        for param in agent_2.encoder.parameters():
            param.requires_grad = True

        return total_loss / epochs