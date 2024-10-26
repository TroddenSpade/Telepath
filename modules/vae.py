import torch
import torch.nn as nn
import torch.nn.functional as F
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


# class Discriminator(nn.Module):
#     def __init__(self, input_shape):
#         super(Discriminator, self).__init__()
#         channels, height, width = input_shape
#         # Calculate output of image discriminator (PatchGAN)
#         self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)

#         def discriminator_block(in_filters, out_filters, normalize=True):
#             """Returns downsampling layers of each discriminator block"""
#             layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
#             if normalize:
#                 layers.append(nn.InstanceNorm2d(out_filters))
#             layers.append(nn.LeakyReLU(0.2, inplace=True))
#             return layers

#         self.model = nn.Sequential(
#             *discriminator_block(channels, 64, normalize=False),
#             *discriminator_block(64, 128),
#             *discriminator_block(128, 256),
#             *discriminator_block(256, 512),
#             nn.Conv2d(512, 1, 3, padding=1)
#         )

#     def forward(self, img):
#         return self.model(img)

class CycleVAE(nn.Module):
    def __init__(self, input_size, latent_size):
        super().__init__()
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

        self.apply(weights_init_normal)


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


    def forward(self, args, agent, agent_2, a_1, a_2, observations, observations_2):
        x_1 = agent.encoder(observations)
        z_1 = self.encoder_1(x_1)
        # mu_1, logvar_1 = self.fc_mu_1(h_1), self.fc_logvar_1(h_1)
        # z_1 = self.reparameterize(mu_1, logvar_1)
        recon_1 = self.decoder_1(z_1)
        belief, _, _, _, posterior_state, _, _ = agent.transition_model(
          torch.zeros(1, args.state_size, device=args.device),
          a_1,
          torch.zeros(1, args.belief_size, device=args.device),
          recon_1.unsqueeze(dim=0))
        belief, posterior_state = belief.squeeze(dim=0), posterior_state.squeeze(dim=0)
        reconst_observation = agent.observation_model(belief, posterior_state)

        x_2 = agent_2.encoder(observations_2)
        z_2 = self.encoder_2(x_2)
        # mu_2, logvar_2 = self.fc_mu_2(h_2), self.fc_logvar_2(h_2)
        # z_2 = self.reparameterize(mu_2, logvar_2)
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
        z_2_ = self.encoder_2(recon_2_from_1_)
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
        z_1_ = self.encoder_1(recon_1_from_2_)
        recon_1_from_2 = self.decoder_2(z_1_)
        belief, _, _, _, posterior_state, _, _ = agent_2.transition_model(
            torch.zeros(1, args.state_size, device=args.device),
            a_2,
            torch.zeros(1, args.belief_size, device=args.device),
            recon_1_from_2.unsqueeze(dim=0))
        belief, posterior_state = belief.squeeze(dim=0), posterior_state.squeeze(dim=0)
        reconst_observation_2_from_1 = agent_2.observation_model(belief, posterior_state)
        
        return reconst_observation, reconst_observation_2, reconst_observation_1_from_2, reconst_observation_2_from_1

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


    def update_parameters(self, epochs, agent, agent_2, D, D_2, optimizer, args, episode):
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
            observations, a_1, _, _ = D.sample(1, 1)
            observations_2, a_2, _, _ = D_2.sample(1, 1)
        
            reconst_observation, reconst_observation_2, reconst_observation_1_from_2, reconst_observation_2_from_1 = self.forward(
            args, agent, agent_2, a_1, a_2, observations[0], observations_2[0])

            recon_loss_1 = CycleVAE.reconstruction_loss(observations[0], reconst_observation)
            recon_loss_2 = CycleVAE.reconstruction_loss(observations_2[0], reconst_observation_2)
            # kl_loss_1 = kl_divergence(mu_1, logvar_1)
            # kl_loss_2 = kl_divergence(mu_2, logvar_2)
            cycle_loss = CycleVAE.cycle_consistency_loss(observations[0], reconst_observation_1_from_2, observations_2[0], reconst_observation_2_from_1)

            loss = (1.0 * (recon_loss_1 + recon_loss_2) +
                        # 0.01 * (kl_loss_1 + kl_loss_2) +
                        5.0 * cycle_loss)
            
            if i % 100 == 99:
                fig, ax = plt.subplots(2, 2, figsize=(2, 2))
                ax[0,0].imshow(observations[0, 0].permute(1, 2, 0).detach().cpu().numpy()+0.5)
                ax[0,1].imshow(reconst_observation[0].permute(1, 2, 0).detach().cpu().numpy()+0.5)
                ax[1,0].imshow(observations_2[0, 0].permute(1, 2, 0).detach().cpu().numpy()+0.5)
                ax[1,1].imshow(reconst_observation_2[0].permute(1, 2, 0).detach().cpu().numpy()+0.5)
                fig.savefig('./results/RRR' + str(episode) + ".png")
                plt.close()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += np.array([recon_loss_1.item(), recon_loss_2.item(), cycle_loss.item()])

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