import argparse
import os
import copy
import numpy as np
import wandb
import torch
from torch.nn import functional as F
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from env import CONTROL_SUITE_ENVS, Env, GYM_ENVS, EnvBatcher
from agent import Dreamer
from memory import ExperienceReplay
from utils import check_power_plugged, lineplot, write_video

# Hyperparameters
parser = argparse.ArgumentParser(description='Dreamer')

parser.add_argument('--seed', type=int, default=1, metavar='S', help='Random seed')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('--env', type=str, default='cheetah-run', choices=GYM_ENVS + CONTROL_SUITE_ENVS, help='Gym/Control Suite environment')
parser.add_argument('--symbolic', action='store_true', help='Symbolic features')
parser.add_argument('--max-episode-length', type=int, default=1000, metavar='T', help='Max episode length')
parser.add_argument('--experience-size', type=int, default=1000000, metavar='D', help='Experience replay size')  # Original implementation has an unlimited buffer size, but 1 million is the max experience collected anyway
parser.add_argument('--cnn-act', type=str, default='relu', choices=dir(F), help='Model activation function for a convolution layer')
parser.add_argument('--dense-act', type=str, default='elu', choices=dir(F), help='Model activation function a dense layer')
parser.add_argument('--embedding-size', type=int, default=1024, metavar='E', help='Observation embedding size')  # Note that the default encoder for visual observations outputs a 1024D vector; for other embedding sizes an additional fully-connected layer is used
parser.add_argument('--hidden-size', type=int, default=300, metavar='H', help='Hidden size')  # paper:300; tf_implementation:400; aligned wit paper. 
parser.add_argument('--belief-size', type=int, default=200, metavar='H', help='Belief/hidden size')
parser.add_argument('--state-size', type=int, default=30, metavar='Z', help='State/latent size')
parser.add_argument('--action-repeat', type=int, default=2, metavar='R', help='Action repeat')

parser.add_argument('--second-action-repeat', type=int, default=4, metavar='R', help='Action repeat')
parser.add_argument('--belief-prior-range', type=int, default=10, metavar='I', help='initial range')
parser.add_argument('--belief-prior-len', type=int, default=8, metavar='I', help='initial length to get the belief prior')
parser.add_argument('--target-horizon', type=int, default=10, metavar='T', help='target horizon')
parser.add_argument('--source-len', type=int, default=5, metavar='S', help='source length')
parser.add_argument('--delay-cem', type=int, default=0, metavar='D', help='delay cem')
parser.add_argument('--expl-amount-2', type=float, default=0.5, help='exploration noise')
parser.add_argument('--second-agenr-lr-skip', type=int, default=5, metavar='S', help='second agent learning rate skip')
parser.add_argument('--check-power-plugged', type=bool, default=True, metavar='C', help='check power plugged')
parser.add_argument('--prior_belief_max_skip', type=int, default=3, metavar='S', help='prior belief max skip')

parser.add_argument('--episodes', type=int, default=1000, metavar='E', help='Total number of episodes')
parser.add_argument('--seed-episodes', type=int, default=5, metavar='S', help='Seed episodes')
parser.add_argument('--collect-interval', type=int, default=100, metavar='C', help='Collect interval')
parser.add_argument('--batch-size', type=int, default=50, metavar='B', help='Batch size')
parser.add_argument('--chunk-size', type=int, default=50, metavar='L', help='Chunk size')
parser.add_argument('--free-nats', type=float, default=3, metavar='F', help='Free nats')
parser.add_argument('--bit-depth', type=int, default=5, metavar='B', help='Image bit depth (quantisation)')
parser.add_argument('--reward_scale', type=float, default=5.0, help='coefficiency term of reward loss')
parser.add_argument('--pcont_scale', type=float, default=5.0, help='coefficiency term of pcont loss')
parser.add_argument('--world_lr', type=float, default=6e-4, metavar='α', help='Learning rate') 
parser.add_argument('--actor_lr', type=float, default=8e-5, metavar='α', help='Learning rate') 
parser.add_argument('--value_lr', type=float, default=8e-5, metavar='α', help='Learning rate') 
parser.add_argument('--learning-rate-schedule', type=int, default=0, metavar='αS', help='Linear learning rate schedule (optimisation steps from 0 to final learning rate; 0 to disable)') 
parser.add_argument('--adam-epsilon', type=float, default=1e-7, metavar='ε', help='Adam optimizer epsilon value') 
# Note that original has a linear learning rate decay, but it seems unlikely that this makes a significant difference
parser.add_argument('--grad-clip-norm', type=float, default=100.0, metavar='C', help='Gradient clipping norm')
parser.add_argument('--expl-amount', type=float, default=0.3, help='exploration noise')
parser.add_argument('--planning-horizon', type=int, default=15, metavar='H', help='Planning horizon distance')
parser.add_argument('--discount', type=float, default=0.99, metavar='H', help='Planning horizon distance')
parser.add_argument('--disclam', type=float, default=0.95, metavar='H', help='discount rate to compute return')
parser.add_argument('--test', action='store_true', help='Test only')
parser.add_argument('--test-interval', type=int, default=25, metavar='I', help='Test interval (episodes)')
parser.add_argument('--test-episodes', type=int, default=10, metavar='E', help='Number of test episodes')
parser.add_argument('--checkpoint-interval', type=int, default=100, metavar='I', help='Checkpoint interval (episodes)')
parser.add_argument('--checkpoint-experience', action='store_true', help='Checkpoint experience replay')
parser.add_argument('--models', type=str, default='', metavar='M', help='Load model checkpoint')
parser.add_argument('--experience-replay', type=str, default='', metavar='ER', help='Load experience replay')
parser.add_argument('--render', action='store_true', help='Render environment')
parser.add_argument('--pcont', action='store_true', help="use the pcont to predict the continuity")
parser.add_argument('--with_logprob', action='store_true', help='use the entropy regularization')
args = parser.parse_args()

wandb.login()
run = wandb.init(project="telepath", config=args)
log_labels = ["observation_loss", "reward_loss", "kl_loss", "pcont_loss", "actor_loss", "value_loss", 'belief_loss']

print(' ' * 26 + 'Options')
for k, v in vars(args).items():
  print(' ' * 26 + k + ': ' + str(v))

# Setup
results_dir = os.path.join('results', args.env, str(args.seed))
os.makedirs(results_dir, exist_ok=True)

# set seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)

if torch.cuda.is_available() and not args.disable_cuda:
  args.device = torch.device('cuda')
  torch.cuda.manual_seed(args.seed)
else:
  args.device = torch.device('cpu')

metrics = {'env_steps': [], 'episodes': [], 'train_rewards': [], 'test_episodes': [], 'test_rewards': [],
           'observation_loss': [], 'reward_loss': [], 'kl_loss': [], 'pcont_loss': [], 'actor_loss': [], 'value_loss': []}

summary_name = results_dir + "/{}_{}_log"

# Initialise training environment and experience replay memory
env = Env(args.env, args.symbolic, args.seed, args.max_episode_length, args.action_repeat, args.bit_depth)
env_2 = Env(args.env, args.symbolic, args.seed, args.max_episode_length, args.second_action_repeat, args.bit_depth)

args.observation_size, args.action_size = env.observation_size, env.action_size
args_2 = copy.deepcopy(args)
args_2.expl_amount = args.expl_amount_2

# Initialise agent
agent = Dreamer(args, is_translation_model=True, wandb_run=run)
agent_2 = Dreamer(args_2, is_translation_model=False)

D = ExperienceReplay(args.experience_size, args.symbolic, env.observation_size, env.action_size, args.bit_depth,
                     args.device)
D_2 = ExperienceReplay(args.experience_size, args.symbolic, env.observation_size, env.action_size, args.bit_depth,
                      args.device)

# Initialise dataset D with S random seed episodes
for s in range(1, args.seed_episodes + 1):
  observation, done, t = env.reset(), False, 0
  while not done:
    action = env.sample_random_action()
    next_observation, reward, done = env.step(action)
    D.append(next_observation, action.cpu(), reward, done)  # here use the next_observation
    observation = next_observation
    t += 1
  metrics['env_steps'].append(t * args.action_repeat + (0 if len(metrics['env_steps']) == 0 else metrics['env_steps'][-1]))
  metrics['episodes'].append(s)
  print("(random)episodes: {}, total_env_steps: {} ".format(metrics['episodes'][-1], metrics['env_steps'][-1]))

for s in range(1, args.seed_episodes + 1):
  observation, done, t = env_2.reset(), False, 0
  while not done:
    action = env_2.sample_random_action()
    next_observation, reward, done = env_2.step(action)
    D_2.append(next_observation, action.cpu(), reward, done)  # here use the next_observation
    observation = next_observation
    t += 1
  print("(actor 2)(random)episodes: {}".format(s))

print("--- Finish random data collection  --- ")

if args.models and os.path.exists(args.models):
  model_dicts = torch.load(args.models)
  agent.transition_model.load_state_dict(model_dicts['transition_model'])
  agent.observation_model.load_state_dict(model_dicts['observation_model'])
  agent.reward_model.load_state_dict(model_dicts['reward_model'])
  agent.encoder.load_state_dict(model_dicts['encoder'])
  agent.actor_model.load_state_dict(model_dicts['actor_model'])
  agent.value_model.load_state_dict(model_dicts['value_model'])
  agent.world_optimizer.load_state_dict(model_dicts['world_optimizer'])
  agent.actor_optimizer.load_state_dict(model_dicts['actor_optimizer'])
  agent.value_optimizer.load_state_dict(model_dicts['value_optimizer'])

# Testing only
if args.test:
  # Set models to eval mode
  agent.transition_model.eval()
  agent.observation_model.eval()
  agent.reward_model.eval()
  agent.encoder.eval()
  agent.actor_model.eval()
  agent.value_model.eval()

  with torch.no_grad():
    total_reward = 0
    for _ in tqdm(range(args.test_episodes)):
      observation = env.reset()

      belief = torch.zeros(1, args.belief_size, device=args.device)
      posterior_state = torch.zeros(1, args.state_size, device=args.device)
      action = torch.zeros(1, env.action_size, device=args.device)

      pbar = tqdm(range(args.max_episode_length // args.action_repeat))
      for t in pbar:
        belief, posterior_state = agent.infer_state(observation.to(device=args.device), action, belief, posterior_state)
        action = agent.select_action((belief, posterior_state), deterministic=True)
        # interact with env
        next_observation, reward, done = env.step(action.cpu() if isinstance(env, EnvBatcher) else action[
          0].cpu())  # Perform environment step (action repeats handled internally)
        total_reward += reward

        observation = next_observation

        if args.render:
          env.render()
        if done:
          pbar.close()
          break

  print('Average Reward:', total_reward / args.test_episodes)
  env.close()
  quit()

# Training (and testing)
for episode in tqdm(range(metrics['episodes'][-1] + 1, args.episodes + 1), total=args.episodes,
                    initial=metrics['episodes'][-1] + 1):
  
  check_power_plugged(args.check_power_plugged)
  
  run.log({"episode": episode}, step=episode)
  data = D.sample(args.batch_size, args.chunk_size)
  data_2 = D_2.sample(args.batch_size, args.chunk_size)
  # Model fitting
  loss_info = agent.train_fn(data, data_2, args.collect_interval, episode)
  print("A1", loss_info)
  run.log(dict(map(lambda i,j : ("env_1/"+i, j) , log_labels, loss_info)), step=episode)
  if episode % args.second_agenr_lr_skip == args.second_agenr_lr_skip - 1:
    loss_info = agent_2.train_fn(data_2, args.collect_interval)
    print("A2", loss_info)
    run.log(dict(map(lambda i,j : ("env_2/"+i, j) , log_labels[:-1], loss_info)), step=episode)
            
  # Data collection
  with torch.no_grad():
    observation, total_reward = env.reset(), 0
    belief = torch.zeros(1, args.belief_size, device=args.device)
    posterior_state = torch.zeros(1, args.state_size, device=args.device)
    action = torch.zeros(1, env.action_size, device=args.device)

    pbar = tqdm(range(args.max_episode_length // args.action_repeat), leave=False, position=0)
    for t in pbar:
      # maintain belief and posterior_state
      belief, posterior_state = agent.infer_state(observation.to(device=args.device), action, belief, posterior_state)
      action = agent.select_action((belief, posterior_state), deterministic=False)

      # interact with env
      next_observation, reward, done = env.step(action.cpu() if isinstance(env, EnvBatcher) else action[0].cpu())  # Perform environment step (action repeats handled internally)
      D.append(next_observation, action.cpu(), reward, done)
      total_reward += reward
      observation = next_observation

      if args.render:
        env.render()
      if done:
        pbar.close()
        break

    # Update and plot train reward metrics
    metrics['env_steps'].append(t * args.action_repeat + metrics['env_steps'][-1])
    metrics['episodes'].append(episode)
    metrics['train_rewards'].append(total_reward)
    lineplot(metrics['episodes'][-len(metrics['train_rewards']):], metrics['train_rewards'], 'train_rewards',
             results_dir)
    print('episode', episode, 'R:', total_reward)
    run.log({"env_1/train_reward": total_reward}, step=episode)

  # Update agent2
  with torch.no_grad():
    observation, total_reward = env_2.reset(), 0
    belief = torch.zeros(1, args.belief_size, device=args.device)
    posterior_state = torch.zeros(1, args.state_size, device=args.device)
    action = torch.zeros(1, env_2.action_size, device=args.device)

    pbar = tqdm(range(args.max_episode_length // args.second_action_repeat), leave=False, position=0)
    for t in pbar:
      # maintain belief and posterior_state
      belief, posterior_state = agent_2.infer_state(observation.to(device=args.device), action, belief, posterior_state)
      action = agent_2.select_action((belief, posterior_state), deterministic=False)

      # interact with env
      next_observation, reward, done = env_2.step(action.cpu() if isinstance(env, EnvBatcher) else action[0].cpu())  # Perform environment step (action repeats handled internally)
      D_2.append(next_observation, action.cpu(), reward, done)
      total_reward += reward
      done += done

      observation = next_observation
      if done:
        pbar.close()
        break
    print('Env 2 - episode', episode, 'R:', total_reward)
    run.log({"env_2/train_reward": total_reward}, step=episode)

  # Test model
  if episode % args.test_interval == 0:
    # Set models to eval mode
    agent.transition_model.eval()
    agent.observation_model.eval()
    agent.reward_model.eval()
    agent.encoder.eval()
    agent.actor_model.eval()
    agent.value_model.eval()

    # Initialise parallelised test environments
    test_envs = EnvBatcher(
      Env,
      (args.env, args.symbolic, args.seed, args.max_episode_length, args.action_repeat, args.bit_depth),
      {},
      args.test_episodes)

    with torch.no_grad():
      observation = test_envs.reset()
      total_rewards = np.zeros((args.test_episodes, ))
      video_frames = []

      belief = torch.zeros(args.test_episodes, args.belief_size, device=args.device)
      posterior_state = torch.zeros(args.test_episodes, args.state_size, device=args.device)
      action = torch.zeros(args.test_episodes, env.action_size, device=args.device)

      for t in tqdm(range(args.max_episode_length // args.action_repeat), leave=False, position=0):
        belief, posterior_state = agent.infer_state(observation.to(device=args.device), action, belief, posterior_state)
        action = agent.select_action((belief, posterior_state), deterministic=True)
        # interact with env
        next_observation, reward, done = test_envs.step(action.cpu() if isinstance(test_envs, EnvBatcher) else action[
          0].cpu())  # Perform environment step (action repeats handled internally)
        total_rewards += reward.numpy()

        if not args.symbolic:  # Collect real vs. predicted frames for video
          video_frames.append(
            make_grid(torch.cat([observation, agent.observation_model(belief, posterior_state).cpu()], dim=3) + 0.5,
                      nrow=5).numpy())  # Decentre
        observation = next_observation
        if done.sum().item() == args.test_episodes:
          pbar.close()
          break


    # Update and plot reward metrics (and write video if applicable) and save metrics
    metrics['test_episodes'].append(episode)
    # metrics['test_rewards'].append(total_rewards.tolist())
    metrics['test_rewards'].append(total_rewards)
    lineplot(metrics['test_episodes'], metrics['test_rewards'], 'test_rewards', results_dir)
    lineplot(np.asarray(metrics['env_steps'])[np.asarray(metrics['test_episodes']) - 1], metrics['test_rewards'],
             'test_rewards_steps', results_dir, xaxis='env_step')
    if not args.symbolic:
      episode_str = str(episode).zfill(len(str(args.episodes)))
      write_video(video_frames, 'test_episode_%s' % episode_str, results_dir)  # Lossy compression
      save_image(torch.as_tensor(video_frames[-1]), os.path.join(results_dir, 'test_episode_%s.png' % episode_str))
    torch.save(metrics, os.path.join(results_dir, 'metrics.pth'))

    # Set models to train mode
    agent.transition_model.train()
    agent.observation_model.train()
    agent.reward_model.train()
    agent.encoder.train()
    agent.actor_model.train()
    agent.value_model.train()
    # Close test environments
    test_envs.close()

  # print("episodes: {}, total_env_steps: {}, train_reward: {} ".format(metrics['episodes'][-1], metrics['env_steps'][-1], metrics['train_rewards'][-1]))
  print()

  # Checkpoint models
  if episode % args.checkpoint_interval == 0:
    torch.save({'transition_model': agent.transition_model.state_dict(),
                'observation_model': agent.observation_model.state_dict(),
                'reward_model1': agent.reward_model.state_dict(),
                'encoder': agent.encoder.state_dict(),
                'actor_model': agent.actor_model.state_dict(),
                'value_model1': agent.value_model.state_dict(),
                'world_optimizer': agent.world_optimizer.state_dict(),
                'actor_optimizer': agent.actor_optimizer.state_dict(),
                'value_optimizer': agent.value_optimizer.state_dict()
                }, os.path.join(results_dir, 'models_%d.pth' % episode))
    if args.checkpoint_experience:
      torch.save(D, os.path.join(results_dir,
                                 'experience_{}.pth'.format(args.env)))  # Warning: will fail with MemoryError with large memory sizes

# Close training environment
env.close()
