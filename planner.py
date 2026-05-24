"""
Planner selects the next best action to take given the latent embedding of
predicted observations
"""
import numpy as np
import torch
import torch.nn.functional as F

class Planner:
    def __init__(self, max_iter, n_samples, n_elites, planning_horizon, mu):
        self.max_iter = max_iter
        self.n_samples = n_samples
        self.mu = mu
        self.n_elites = n_elites
        self.horizon = planning_horizon
    
    def objective_function(self, z_H, z_g):
        """
        Computes the squared difference between the predicted final latent embedding 
        at the end of a time horizon (z_H) with the latent embedding of the target (z_g)
        to evaluate if the model reached the goal state
        """
        return F.mse_loss(z_H, z_g)

    def planner(self, lewm, obs, obs_goal, action_dim):
        """
        Samples and selects the best action sequence to take given a single
        current observation and goal observation.
        - Cross Entropy Method (CEM): selects 300 candidate action sequences up to time horizon (H)
        - LeWM rollout: actions are applied to current observation to predict the next observation embedding
        - Action selection: the best action seq selected based on the lowest planning cost
        
        Input: current observation and goal observation
        Outputs: (H, A) best action sequence found after max_iter
        """
        # initialize mu and sigma for sampling distribution
        mu = np.zeros((self.horizon, action_dim))
        sigma = np.ones((self.horizon, action_dim))

        # send to gpu if available
        device = torch.device("cuda" if torch.cuda.is_available() else "mps")
        obs = obs.to(device)
        obs_goal = obs_goal.to(device)

        # encode the obs and obs_goal
        with torch.no_grad():
            lewm.eval()
            z1 = lewm.encoder(obs)
            zg = lewm.encoder(obs_goal) # ex: tree

        for _ in range(self.max_iter):
            # 1. Action sampling: sample 300 candidate action samples (n_samples)
            #   each sample contains action sequences up to time horizon (H)
            samples = np.zeros((self.n_samples, self.horizon, action_dim))
            samples[..., :8] = np.random.normal(mu[:, :8], sigma[:, :8], size=(self.n_samples, self.horizon, 8)) # binary actions
            samples[..., :8] = (samples[..., :8] > 0.5).astype(np.float32) # binarize
            samples[..., 8:] = np.random.normal(mu[:, 8:], sigma[:, 8:], size=(self.n_samples, self.horizon, 2)) # camera

            scores = []
            for actions in samples:
                # 2. Rollout actions in world model (imagination)
                #   predict forward H steps with this sample's actions
                z_pred = z1.unsqueeze(0).unsqueeze(0) # [1, 1, 192]
                for t in range(self.horizon):
                    a_t = torch.as_tensor(actions[t], dtype=torch.float32, device=device).view(1, 1, -1)
                    a_emb = lewm.action_embedder(a_t)
                    next_emb = lewm.predictor(z_pred, a_emb)
                    z_pred = next_emb[:, -1:, :]
                z_pred = z_pred[:, -1, :] # Final predicted obs at end of horizon

                # 3. Compute cost: how close imagined final state is to fixed goal zg
                # cost should decrease over time as the model moves closer to the goal
                score = self.objective_function(z_pred, zg)
                scores.append(score.item())

            #  update distribution parameters based on elites
            elite_idx = np.argsort(scores)[:self.n_elites] # top n_elites
            elites = samples[elite_idx]

            mu = elites.mean(axis=0)
            sigma = elites.std(axis=0) + 1e-6

        # 4. Action selection: return best action sequence found after max_iter
        return samples[np.argmin(scores)]
