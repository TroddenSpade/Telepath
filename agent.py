import os
from copy import deepcopy
import time
import cv2
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.distributions import Normal
from torch.distributions.independent import Independent
from torch.distributions.kl import kl_divergence
from torch.nn import functional as F
from tqdm import tqdm
from memory import ExperienceReplay
from models import ApproxActionModel, bottle, Encoder, ObservationModel, RewardModel, ScriptedRSSM, ValueModel, ActorModel, PCONTModel

from modules import CEM
from modules.skill import LSTMBeliefPrior


def cal_returns(reward, value, bootstrap, pcont, lambda_):
    """
    Calculate the target value, following equation (5-6) in Dreamer
    :param reward, value: imagined rewards and values, dim=[horizon, (chuck-1)*batch, reward/value_shape]
    :param bootstrap: the last predicted value, dim=[(chuck-1)*batch, 1(value_dim)]
    :param pcont: gamma
    :param lambda_: lambda
    :return: the target value, dim=[horizon, (chuck-1)*batch, value_shape]
    """
    assert list(reward.shape) == list(
        value.shape), "The shape of reward and value should be similar"
    if isinstance(pcont, (int, float)):
        pcont = pcont * torch.ones_like(reward)

    # bootstrap[None] is used to extend additional dim
    next_value = torch.cat((value[1:], bootstrap[None]), 0)
    inputs = reward + pcont * next_value * \
        (1 - lambda_)  # dim=[horizon, (chuck-1)*B, 1]
    outputs = []
    last = bootstrap

    for t in reversed(range(reward.shape[0])):  # for t in horizon
        inp = inputs[t]
        last = inp + pcont[t] * lambda_ * last
        outputs.append(last)
    returns = torch.flip(torch.stack(outputs), [0])
    return returns


def count_vars(module):
    """ count parameters number of module"""
    return sum([np.prod(p.shape) for p in module.parameters()])


class Dreamer():
    def __init__(self, args, is_translation_model=False):
        """
        All paras are passed by args
        :param args: a dict that includes parameters
        """
        super().__init__()
        self.args = args

        if is_translation_model:
            self.train_fn = self.update_telepath_parameters
        else:
            self.train_fn = self.update_dreamer_parameters

        self.cem = CEM(10, 128, 1/8., args.action_size)

        self.embedding_encoder = LSTMBeliefPrior(
            args.embedding_size,
            args.hidden_size,
            args.belief_size).to(device=args.device)

        # Initialise model parameters randomly
        self.transition_model = ScriptedRSSM(
            args.belief_size,
            args.state_size,
            args.action_size,
            args.hidden_size,
            args.embedding_size,
            args.dense_act).to(device=args.device)

        self.observation_model = ObservationModel(
            args.symbolic,
            args.observation_size,
            args.belief_size,
            args.state_size,
            args.embedding_size,
            activation_function=(args.dense_act if args.symbolic else args.cnn_act)).to(device=args.device)

        self.reward_model = RewardModel(
            args.belief_size,
            args.state_size,
            args.hidden_size,
            args.dense_act).to(device=args.device)

        self.approx_action = ApproxActionModel(
            args.embedding_size,
            args.action_size,
            args.hidden_size).to(device=args.device)

        self.encoder = Encoder(
            args.symbolic,
            args.observation_size,
            args.embedding_size,
            args.cnn_act).to(device=args.device)

        self.actor_model = ActorModel(
            args.action_size,
            args.belief_size,
            args.state_size,
            args.hidden_size,
            activation_function=args.dense_act).to(device=args.device)

        self.value_model = ValueModel(
            args.belief_size,
            args.state_size,
            args.hidden_size,
            args.dense_act).to(device=args.device)

        self.pcont_model = PCONTModel(
            args.belief_size,
            args.state_size,
            args.hidden_size,
            args.dense_act).to(device=args.device)

        self.target_value_model = deepcopy(self.value_model)

        for p in self.target_value_model.parameters():
            p.requires_grad = False

        # setup the paras to update
        self.world_param = list(self.transition_model.parameters())\
            + list(self.observation_model.parameters())\
            + list(self.encoder.parameters())
        if is_translation_model:
            self.world_param += list(self.embedding_encoder.parameters())
        if args.pcont:
            self.world_param += list(self.pcont_model.parameters())

        # setup optimizer
        self.world_optimizer = optim.Adam(self.world_param, lr=args.world_lr)
        self.actor_optimizer = optim.Adam(
            self.actor_model.parameters(), lr=args.actor_lr)
        self.value_optimizer = optim.Adam(
            list(self.value_model.parameters()), lr=args.value_lr)
        self.reward_optimizer = optim.Adam(
            list(self.reward_model.parameters()), lr=args.world_lr)
        # self.bp_optimizer = optim.Adam(
        #     list(self.embedding_encoder.parameters()), lr=args.world_lr)

        # setup the free_nat
        # Allowed deviation in KL divergence
        self.free_nats = torch.full(
            (1, ), args.free_nats, dtype=torch.float32, device=args.device)
        self.nll_loss = nn.GaussianNLLLoss()

    def process_im(self, image):
        # Resize, put channel first, convert it to a tensor, centre it to [-0.5, 0.5] and add batch dimenstion.

        def preprocess_observation_(observation, bit_depth):
            # Preprocesses an observation inplace (from float32 Tensor [0, 255] to [-0.5, 0.5])
            observation.div_(2 ** (8 - bit_depth)).floor_().div_(2 ** bit_depth).sub_(
                0.5)  # Quantise to given bit depth and centre
            observation.add_(torch.rand_like(observation).div_(
                2 ** bit_depth))  # Dequantise (to approx. match likelihood of PDF of continuous images vs. PMF of discrete images)

        image = torch.tensor(cv2.resize(image, (64, 64), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1),
                             dtype=torch.float32)  # Resize and put channel first

        preprocess_observation_(image, self.args.bit_depth)
        return image.unsqueeze(dim=0)

    def _compute_loss_world(self, state, data):
        # unpackage data
        beliefs, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs = state
        observations, rewards, nonterminals = data

        observation_loss = F.mse_loss(
            bottle(self.observation_model, (beliefs, posterior_states)),
            observations,
            reduction='none').sum(dim=2 if self.args.symbolic else (2, 3, 4)).mean(dim=(0, 1))

        # transition loss
        kl_loss = torch.max(
            kl_divergence(
                Independent(Normal(posterior_means, posterior_std_devs), 1),
                Independent(Normal(prior_means, prior_std_devs), 1)),
            self.free_nats).mean(dim=(0, 1))

        if self.args.pcont:
            pcont_loss = F.binary_cross_entropy(
                bottle(self.pcont_model, (beliefs, posterior_states)), nonterminals)

        if rewards is not None:
            reward_loss = F.mse_loss(
                bottle(self.reward_model, (beliefs, posterior_states)),
                rewards,
                reduction='none').mean(dim=(0,1))  # TODO: 5
            return observation_loss, self.args.reward_scale * reward_loss, kl_loss, (self.args.pcont_scale * pcont_loss if self.args.pcont else 0)
        else:
            return observation_loss, kl_loss, (self.args.pcont_scale * pcont_loss if self.args.pcont else 0)


    def _compute_loss_actor(self, imag_beliefs, imag_states, imag_ac_logps=None):
        # reward and value prediction of imagined trajectories
        imag_rewards = bottle(self.reward_model, (imag_beliefs, imag_states))
        imag_values = bottle(self.value_model, (imag_beliefs, imag_states))

        with torch.no_grad():
            if self.args.pcont:
                pcont = bottle(self.pcont_model, (imag_beliefs, imag_states))
            else:
                pcont = self.args.discount * torch.ones_like(imag_rewards)
        pcont = pcont.detach()

        if imag_ac_logps is not None:
            imag_values[1:] -= self.args.temp * \
                imag_ac_logps  # add entropy here

        returns = cal_returns(
            imag_rewards[:-1], imag_values[:-1], imag_values[-1], pcont[:-1], lambda_=self.args.disclam)

        discount = torch.cumprod(
            torch.cat([torch.ones_like(pcont[:1]), pcont[:-2]], 0), 0).detach()

        actor_loss = -torch.mean(discount * returns)
        return actor_loss


    def _compute_loss_critic(self, imag_beliefs, imag_states, imag_ac_logps=None):
        with torch.no_grad():
            # calculate the target with the target nn
            target_imag_values = bottle(
                self.target_value_model, (imag_beliefs, imag_states))
            imag_rewards = bottle(
                self.reward_model, (imag_beliefs, imag_states))

            if self.args.pcont:
                pcont = bottle(self.pcont_model, (imag_beliefs, imag_states))
            else:
                pcont = self.args.discount * torch.ones_like(imag_rewards)

            if imag_ac_logps is not None:
                target_imag_values[1:] -= self.args.temp * imag_ac_logps

        returns = cal_returns(imag_rewards[:-1], target_imag_values[:-1],
                              target_imag_values[-1], pcont[:-1], lambda_=self.args.disclam)
        target_return = returns.detach()

        value_pred = bottle(self.value_model, (imag_beliefs, imag_states))[:-1]

        value_loss = F.mse_loss(value_pred, target_return,
                                reduction="none").mean(dim=(0, 1))

        return value_loss


    def _latent_imagination(self, beliefs, posterior_states, with_logprob=False):
        # Rollout to generate imagined trajectories
        chunk_size, batch_size, _ = list(
            posterior_states.size())  # flatten the tensor
        flatten_size = chunk_size * batch_size

        posterior_states = posterior_states.detach().reshape(flatten_size, -1)
        beliefs = beliefs.detach().reshape(flatten_size, -1)

        imag_beliefs, imag_states, imag_ac_logps = [
            beliefs], [posterior_states], []

        for i in range(self.args.planning_horizon):
            imag_action, imag_ac_logp = self.actor_model(
                imag_beliefs[-1].detach(),
                imag_states[-1].detach(),
                deterministic=False,
                with_logprob=with_logprob,
            )
            imag_action = imag_action.unsqueeze(dim=0)  # add time dim

            imag_belief, imag_state, _, _ = self.transition_model(
                imag_states[-1], imag_action, imag_beliefs[-1])
            imag_beliefs.append(imag_belief.squeeze(dim=0))
            imag_states.append(imag_state.squeeze(dim=0))

            if with_logprob:
                imag_ac_logps.append(imag_ac_logp.squeeze(dim=0))

        # shape [horizon+1, (chuck-1)*batch, belief_size]
        imag_beliefs = torch.stack(imag_beliefs, dim=0).to(self.args.device)
        imag_states = torch.stack(imag_states, dim=0).to(self.args.device)

        if with_logprob:
            imag_ac_logps = torch.stack(imag_ac_logps, dim=0).to(
                self.args.device)  # shape [horizon, (chuck-1)*batch]

        return imag_beliefs, imag_states, imag_ac_logps if with_logprob else None


    def translate_trajectory(self, target_observations, rewards, source_length, target_horizon):
        batch_size_ = target_observations.size(1)
        sample_length = 10
        with torch.no_grad():
            embeds = bottle(self.encoder, (target_observations[:self.args.belief_prior_len], ))
            means_, stds_, _ = self.embedding_encoder(embeds)
            means_, stds_ = means_[[-1]], stds_[[-1]]
            prior_beliefs = means_ + stds_ * torch.randn((sample_length, batch_size_, self.args.belief_size), device=self.args.device)
            prior_states = self.transition_model._posterior_state(
                prior_beliefs.flatten(0,1), 
                embeds[-1].repeat(sample_length, 1)
            ).unflatten(0, (sample_length, batch_size_))

        translated_beliefs, translated_states, translated_rewards = [torch.empty(0)] * batch_size_, \
                                                                    [torch.empty(0)] * batch_size_, \
                                                                    [torch.empty(0)] * batch_size_

        for i in tqdm(range(batch_size_), leave=False, position=0, desc="CEM training"):
            translated_beliefs[i], translated_states[i], translated_rewards[i] = self.cem.train(
                prior_beliefs[:, i],
                prior_states[:, i],
                target_observations[self.args.belief_prior_len:source_length+self.args.belief_prior_len, i],
                rewards[self.args.belief_prior_len:source_length+self.args.belief_prior_len, i],
                target_horizon,
                self.transition_model,
                self.observation_model,
                i == batch_size_-1
            )

        translated_beliefs, translated_states, translated_rewards = torch.stack(translated_beliefs, dim=1), \
            torch.stack(translated_states, dim=1), \
            torch.stack(translated_rewards, dim=1)

        return translated_beliefs, translated_states, translated_rewards


    def update_reward_model(self, gradient_steps, translated_beliefs, translated_states, rewards):
        loss = 0.0
        for i in tqdm(range(gradient_steps), leave=False, position=0, desc="reward training"):
            reward_loss = F.mse_loss(
                bottle(self.reward_model,
                       (translated_beliefs, translated_states)),
                rewards,
                reduction='none').mean(dim=(0, 1)) * self.args.reward_scale
            self.reward_optimizer.zero_grad()
            reward_loss.backward()
            self.reward_optimizer.step()
            loss += reward_loss.item()

        return loss/gradient_steps


    def imag_initial_states(self, initial_length, observations, actions, nonterminals, plot_reconstruction=True):
        with torch.no_grad():
            embds_2 = bottle(self.encoder, (observations[:initial_length], ))
            # actions_2 = bottle(self.approx_action, (embds_2[:-1], embds_2[1:]))
            actions_2 = actions[:initial_length]

            # Update belief/state using posterior from previous belief/state, previous action and current observation (over entire sequence at once)
            beliefs, _, _, _, posterior_states, _, _ = self.transition_model(
                torch.zeros(self.args.batch_size,
                            self.args.state_size, device=self.args.device),
                actions_2,
                torch.zeros(self.args.batch_size,
                            self.args.belief_size, device=self.args.device),
                embds_2,
                nonterminals[:initial_length])

            reconst_observations = bottle(self.observation_model, (beliefs, posterior_states))

        if plot_reconstruction:
            obs = np.clip(observations[:initial_length, 0].cpu().permute(
                0, 2, 3, 1).numpy() + 0.5, 0, 1)
            r_obs = np.clip(reconst_observations[:, 0].cpu().permute(
                0, 2, 3, 1).numpy() + 0.5, 0, 1)
            fig, ax = plt.subplots(2, initial_length, figsize=(16, 4))
            for i in range(initial_length):
                ax[0, i].imshow(obs[i])
                ax[1, i].imshow(r_obs[i])
                ax[0, i].axis("off")
                ax[1, i].axis("off")
            fig.savefig('./results/R-' + str(int(time.time()) % 20) + ".png")
            plt.close()

        return beliefs[-1].detach(), posterior_states[-1].detach()


    # def update_belief_prior(self, belief_means, belief_stds, embeds):
    #     last_idx = (embeds.size(0)-1)
    #     rand_idxs_r = np.array([sorted(set(np.random.choice(np.arange(last_idx), self.args.belief_prior_len-1, replace=False))) for _ in range(embeds.size(1))])
    #     rand_idxs_r = np.concatenate([rand_idxs_r, last_idx*np.ones((embeds.size(1), 1))], axis=1).T
    #     rand_idxs_c = np.repeat(np.arange(embeds.size(1))[:, None], self.args.belief_prior_len, axis=1).T

    #     inputs = embeds[rand_idxs_r, rand_idxs_c]

    #     z_means, z_stds, z = self.embedding_encoder(inputs.detach())

    #     kl_div = torch.max(
    #         kl_divergence(
    #             Independent(Normal(belief_means[last_idx], belief_stds[last_idx]), 1),
    #             Independent(Normal(z_means, z_stds), 1)),
    #             self.free_nats).mean()

    #     return kl_div


    def update_dreamer_parameters(self, data, gradient_steps):
        loss_info = []  # used to record loss

        for s in tqdm(range(gradient_steps), position=0, leave=False):
            # get state and belief of samples
            observations, actions, rewards, nonterminals = data

            init_belief = torch.zeros(self.args.batch_size, self.args.belief_size, device=self.args.device)
            init_state = torch.zeros(self.args.batch_size, self.args.state_size, device=self.args.device)

            # Update belief/state using posterior from previous belief/state, previous action and current observation (over entire sequence at once)
            beliefs, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs = self.transition_model(
                init_state,
                actions,
                init_belief,
                bottle(self.encoder, (observations, )),
                nonterminals)  # TODO: 4

            # update paras of world model
            world_model_loss = self._compute_loss_world(
                state=(beliefs, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs),
                data=(observations, rewards, nonterminals)
            )
            observation_loss, reward_loss, kl_loss, pcont_loss = world_model_loss
            self. world_optimizer.zero_grad()
            (observation_loss + reward_loss + kl_loss + pcont_loss).backward()
            nn.utils.clip_grad_norm_(self.world_param, self.args.grad_clip_norm, norm_type=2)
            self.world_optimizer.step()

            # freeze params to save memory
            for p in self.world_param:
                p.requires_grad = False
            for p in self.value_model.parameters():
                p.requires_grad = False

            # latent imagination
            imag_beliefs, imag_states, imag_ac_logps = self._latent_imagination(beliefs, posterior_states, with_logprob=self.args.with_logprob)

            # update actor
            actor_loss = self._compute_loss_actor(imag_beliefs, imag_states, imag_ac_logps=imag_ac_logps)

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor_model.parameters(), self.args.grad_clip_norm, norm_type=2)
            self.actor_optimizer.step()

            for p in self.world_param:
                p.requires_grad = True
            for p in self.value_model.parameters():
                p.requires_grad = True

            # update critic
            imag_beliefs = imag_beliefs.detach()
            imag_states = imag_states.detach()

            critic_loss = self._compute_loss_critic(imag_beliefs, imag_states, imag_ac_logps=imag_ac_logps)

            self.value_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.value_model.parameters(), self.args.grad_clip_norm, norm_type=2)
            self.value_optimizer.step()

            loss_info.append([observation_loss.item(), reward_loss.item(), kl_loss.item(), pcont_loss.item() if self.args.pcont else 0, actor_loss.item(), critic_loss.item()])

        # finally, update target value function every #gradient_steps
        with torch.no_grad():
            self.target_value_model.load_state_dict(self.value_model.state_dict())

        return np.array(loss_info).mean(0)


    def update_telepath_parameters(self, data, data_2, gradient_steps, global_step):
        loss_info = []  # used to record loss

        ###### Agent World Model Training ######
        for s in tqdm(range(gradient_steps), leave=False, position=0, desc="updating parameters"):
            # get state and belief of samples
            observations, actions, _, nonterminals = data

            init_belief = torch.zeros(
                self.args.batch_size, self.args.belief_size, device=self.args.device)
            init_state = torch.zeros(
                self.args.batch_size, self.args.state_size, device=self.args.device)

            embeds = bottle(self.encoder, (observations, ))
            # Update belief/state using posterior from previous belief/state, previous action and current observation (over entire sequence at once)
            beliefs, prior_states, prior_means, prior_std_devs,\
                    posterior_states, posterior_means, posterior_std_devs = self.transition_model(
                init_state,
                actions,
                init_belief,
                embeds,
                nonterminals)

            max_skip = 3
            beliefs_loss = 0
            for _ in range(max_skip):
                idxs = np.cumsum(np.random.randint(1, max_skip+1, size=self.args.chunk_size))
                idxs = idxs[np.where(idxs<self.args.chunk_size)[0]]

                means_, stds_, z = self.embedding_encoder(embeds[idxs])
                beliefs_loss += self.nll_loss(beliefs[idxs], means_, stds_)

            # update paras of world model
            world_model_loss = self._compute_loss_world(
                state=(beliefs, prior_states, prior_means, prior_std_devs,
                       posterior_states, posterior_means, posterior_std_devs),
                data=(observations, None, nonterminals)
            )
            observation_loss, kl_loss, pcont_loss = world_model_loss
            self.world_optimizer.zero_grad()
            (observation_loss + kl_loss + pcont_loss + beliefs_loss).backward()
            nn.utils.clip_grad_norm_(
                self.world_param, self.args.grad_clip_norm, norm_type=2)
            self.world_optimizer.step()

            # freeze params to save memory
            for p in self.world_param:
                p.requires_grad = False
            for p in self.value_model.parameters():
                p.requires_grad = False

            # latent imagination
            imag_beliefs, imag_states, imag_ac_logps = self._latent_imagination(
                beliefs, posterior_states, with_logprob=self.args.with_logprob)

            # update actor
            actor_loss = self._compute_loss_actor(
                imag_beliefs, imag_states, imag_ac_logps=imag_ac_logps)

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(
                self.actor_model.parameters(), self.args.grad_clip_norm, norm_type=2)
            self.actor_optimizer.step()

            for p in self.world_param:
                p.requires_grad = True
            for p in self.value_model.parameters():
                p.requires_grad = True

            # update critic
            imag_beliefs = imag_beliefs.detach()
            imag_states = imag_states.detach()

            critic_loss = self._compute_loss_critic(
                imag_beliefs, imag_states, imag_ac_logps=imag_ac_logps)

            self.value_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(
                self.value_model.parameters(), self.args.grad_clip_norm, norm_type=2)
            self.value_optimizer.step()

            loss_info.append([
                observation_loss.item(),
                0,
                kl_loss.item(),
                pcont_loss.item() if self.args.pcont else 0,
                actor_loss.item(),
                critic_loss.item(),
                beliefs_loss.item()])

        # finally, update target value function every #gradient_steps
        with torch.no_grad():
            self.target_value_model.load_state_dict(
                self.value_model.state_dict())

        # del imag_beliefs, imag_states, imag_ac_logps
        # del beliefs, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs

        loss_info = np.array(loss_info).mean(0)
        ####### Translation #######
        if global_step > self.args.delay_cem:
            observations_2, _, rewards_2, _ = data_2

            translated_beliefs, translated_states, translated_rewards = self.translate_trajectory(
                observations_2,
                rewards_2,
                source_length=self.args.source_len,
                target_horizon=self.args.target_horizon)

            reward_loss = self.update_reward_model(
                gradient_steps,
                translated_beliefs.detach(),
                translated_states.detach(),
                translated_rewards.detach())

            loss_info[1] = reward_loss

        fig, ax = plt.subplots(1, 2, figsize=(2, 2))
        ax[0].imshow(observations[0, 0].permute(1, 2, 0).cpu().numpy()+0.5)
        ax[1].imshow(observations_2[0, 0].permute(1, 2, 0).cpu().numpy()+0.5)
        fig.savefig('./results/' + str(int(time.time()) % 5) + ".png")
        plt.close()

        return loss_info


    def infer_state(self, observation, action, belief=None, state=None):
        """ Infer belief over current state q(s_t|o≤t,a<t) from the history,
            return updated belief and posterior_state at time t
            returned shape: belief/state [belief/state_dim] (remove the time_dim)
        """
        # observation is obs.to(device), action.shape=[act_dim] (will add time dim inside this fn), belief.shape
        belief, _, _, _, posterior_state, _, _ = self.transition_model(
            state,
            action.unsqueeze(dim=0),
            belief,
            self.encoder(observation).unsqueeze(dim=0))  # Action and observation need extra time dimension

        belief, posterior_state = belief.squeeze(dim=0), posterior_state.squeeze(
            dim=0)  # Remove time dimension from belief/state

        return belief, posterior_state

    def select_action(self, state, deterministic=False):
        # get action with the inputs get from fn: infer_state; return a numpy with shape [batch, act_size]
        belief, posterior_state = state
        action, _ = self.actor_model(
            belief, posterior_state, deterministic=deterministic, with_logprob=False)

        if not deterministic and not self.args.with_logprob:  # add exploration noise
            action = Normal(action, self.args.expl_amount).rsample()
            action = torch.clamp(action, -1, 1)
        return action  # tensor
