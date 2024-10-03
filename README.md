![cheetah-run-4-2](https://github.com/user-attachments/assets/d094cc66-f905-4531-a557-6c067f78d02b)![Walker-Walk-3-1](https://github.com/user-attachments/assets/042fed99-135b-4f13-8b36-c210c2820434)![BallInCup-6-3](https://github.com/user-attachments/assets/5885b06f-2f38-4d46-938b-e58b190c372e)# Mitigating Dynamic Gaps in Policy Transfer Through Trajectory Alignment with Model-based Reinforcement Learning and Dynamic Time Warping

This project presents an approach for transferring policies between two environments characterized by dynamic gaps, where the target environment lacks inherent rewards. The proposed method leverages model-based reinforcement learning (RL), specifically the Dreamer algorithm, to determine the optimal trajectory within the target environment by aligning it with the trajectory from the source environment. Dynamic Time Warping (DTW) is employed as the primary metric for evaluating the similarity between trajectories based solely on observations, excluding any reliance on environments' states or external signals. This allows for effective alignment while accounting for temporal fluctuations and disparities in dynamics between the two environments. Additionally, the approach incorporates a reward calculation mechanism driven by aligning observations from the source environment, facilitating improved policy adaptation in the target domain. This observation-based alignment is crucial, particularly in the absence of rewards in the target environment.

## Dynamics

Model-based reinforcement learning (MBRL) involves the development of a dynamics model that anticipates the subsequent state based on the present state and action, facilitating the simulation of "imaginary" trajectories to enhance sampling efficiency. This is achieved by leveraging interaction experiences through a world model, which acts as reusable, task-agnostic knowledge across different tasks. The Dreamer agent, a notable advancement in MBRL, learns in a latent representation space, achieving superior performance and efficiency. In this project, two Dreamer models are used to understand the dynamics of both source and target environments, enabling effective planning and policy transfer between them.

![Telepath-Page-4 drawio](https://github.com/user-attachments/assets/0946a231-617d-425f-9e0e-4a28c6badcec)

The source environment leverages Dreamer's inherent reward model, predicting rewards by learning directly from the environment's reward signals. Dreamer uses these signals to refine its dynamics model and optimize policy performance within the source domain. In contrast, the target environment lacks any reward signals, necessitating an independent reward model. This model is trained by calculating rewards based on trajectory alignments between the source and target environments, where alignment is achieved through Dynamic Time Warping (DTW) using only observations, allowing for effective policy transfer without external rewards in the target domain.

## Bridging Source and Target Environments

We require the corresponding state of that observation in the target domain to predict subsequent states in the source environment based on an observation using the world model of the target environment. However, direct access to the source environment’s states as defined by the target Dreamer model is impractical, as we only have access to the trajectories from either the source or target environments.

![Telepath-Copy of Page-2 drawio](https://github.com/user-attachments/assets/f8e25ee5-7fab-47f4-9b07-5a05cf4d1e79)

To address this challenge, we propose a temporally invariant prior belief neural network, an extension of the Dreamer model for the target environment's world model. When provided with a sequence of observations, this network generates a distribution over possible states that could represent the state of the final observation in the input set. Specifically, it allows for the prediction of states from a source trajectory that is interpretable by the target world model.

This extension is integrated solely into the target world model and is trained using a randomly selected subset of target trajectories, enhancing its temporal invariance. Once the state corresponding to a specific observation is estimated, we can predict subsequent states and their associated observations. This enables us to generate aligned trajectories in the source and target environments, starting from a common observation.

## Finding the Right Alignment
The trajectories from the source environment are divided into two segments: the first segment is used to estimate an initial state in the target domain, while the second segment is employed to identify an equivalent sequence in the target environment.

Multiple initial states are sampled from the output distribution of the temporally invariant belief network. For each sampled state, we determine the optimal sequence of actions that best aligns with the second segment of the source environment’s trajectory. To identify the optimal set of actions in the target domain, the Cross-Entropy method is applied, optimizing the alignment between the resulting observations and the source trajectory. Dynamic Time Warping (DTW) is utilized to achieve this alignment by minimizing the loss function, thereby tightening the correspondence between the imagined trajectory in the target environment and the original trajectory from the source environment.

![281](https://github.com/user-attachments/assets/121485a1-4278-432e-b48f-df7faa0fd338)

![N-1723261453](https://github.com/user-attachments/assets/2d654c12-d3d9-4d86-8960-84e61d0b73d9)
![N-1723264536](https://github.com/user-attachments/assets/77135d34-ff62-4de3-a2ae-9daffe0d5405)
![N-1723266437](https://github.com/user-attachments/assets/b5ebfcf3-edb7-4bbd-86cc-660c4408a858)
![N-1723276875](https://github.com/user-attachments/assets/a2798c74-4f53-45df-8852-a644d8e577da)
![N-1723281727](https://github.com/user-attachments/assets/be7d24e7-2776-4c22-90bf-ea621068e984)


The rewards for the target model are calculated proportionally by evaluating the ratio of source transitions to target transitions following the alignment. This proportional reward scheme reflects how closely the transitions in the target environment align with those of the source environment, ensuring that the reward signal is informed by the degree of alignment between the two trajectories.

## Results
