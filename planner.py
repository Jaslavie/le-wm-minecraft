"""
Planner selects the next best action to take given the latent embedding of
predicted observations
"""
import numpy as np
import torch
import torch.nn.functional as F
from utils import normalize_camera

class Planner:
    def __init__(self, max_iter, n_samples, n_elites, planning_horizon, action_dim):
        self.max_iter = max_iter
        self.n_samples = n_samples
        self.n_elites = n_elites
        self.horizon = planning_horizon
        self.action_dim = action_dim
    
    def objective_function(self, z_H, z_g):
        """
        Computes the squared difference between the predicted final latent embedding 
        at the end of a time horizon (z_H) with the latent embedding of the target (z_g)
        to evaluate if the model reached the goal state
        """
        return F.mse_loss(z_H, z_g)

    def planner(self, lewm, obs, obs_goal, cam_mean, cam_std):
        """
        Samples and selects the best action sequence to take given a single
        current observation and goal observation.
        - Cross Entropy Method (CEM): selects 300 candidate action sequences up to time horizon (H)
        - LeWM rollout: actions are applied to current observation to predict the next observation embedding
        - Action selection: the best action seq selected based on the lowest planning cost
        
        Input: current observation and goal observation frame
        Outputs: (H, A) best action sequence found after max_iter
        """
        # initialize mu and sigma for sampling distribution
        mu = np.zeros((self.horizon, self.action_dim))
        sigma = np.ones((self.horizon, self.action_dim))

        # send to gpu if available
        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        obs = obs.to(device)
        obs_goal = obs_goal.to(device)
        cam_mean = cam_mean.to(device)
        cam_std = cam_std.to(device)

        # encode the obs and obs_goal
        with torch.no_grad():
            lewm.eval()
            z1 = lewm.encoder(obs) # current frame
            zg = lewm.encoder(obs_goal) # ex: tree

        for _ in range(self.max_iter):
            # 1. Action sampling: sample 300 candidate action samples (n_samples)
            #   each sample contains action sequences up to time horizon (H)
            samples = np.zeros((self.n_samples, self.horizon, self.action_dim))
            samples[..., :8] = np.random.normal(mu[:, :8], sigma[:, :8], size=(self.n_samples, self.horizon, 8)) # binary actions
            samples[..., :8] = (samples[..., :8] > 0.5).astype(np.float32) # binarize
            samples[..., 8:] = np.random.normal(mu[:, 8:], sigma[:, 8:], size=(self.n_samples, self.horizon, 2)) # camera

            scores = []
            for actions in samples:
                # 2. Rollout actions in world model (imagination)
                # store history of past 8 observations and actions. recall that predictor
                # only has access to last 8 actions/obs in its memory
                z_pred_hist = [z1.view(1, 1, -1)]
                a_emb_hist = []
                
                for t in range(self.horizon):
                    # embed and store each sampled action in the imagination horizon
                    a_t = torch.as_tensor(actions[t], dtype=torch.float32, device=device).view(1, 1, -1)
                    a_t = normalize_camera(a_t, cam_mean, cam_std) # normalize camera
                    a_emb = lewm.action_embedder(a_t)
                    a_emb_hist.append(a_emb)
                    
                    # stack current memory to send to predictor
                    obs_context = torch.cat(z_pred_hist, dim=1)
                    act_context = torch.cat(a_emb_hist, dim=1)

                    # add a copy of last obs/emb since predictor removes the last timestamp
                    emb = torch.cat([obs_context, obs_context[:, -1:, :]], dim=1)
                    action_emb = torch.cat([act_context, act_context[:, -1:, :]], dim=1)
                    
                    # run predictor on last observation
                    next_emb = lewm.predictor(emb[:, :-1], action_emb[:, :-1])
                    z_pred = next_emb[:, -1:, :]
                    z_pred_hist.append(z_pred)

                # get final predicted obs at end of horizon   
                z_pred_hist = z_pred_hist[-1].view(-1)

                # 3. Compute cost: how close imagined final state is to fixed goal zg
                # cost should decrease over time as the model moves closer to the goal
                score = self.objective_function(z_pred_hist[-1].view(-1), zg)
                scores.append(score.item())

            #  update distribution parameters based on elites
            elite_idx = np.argsort(scores)[:self.n_elites] # top n_elites
            elites = samples[elite_idx]

            mu = elites.mean(axis=0)
            sigma = elites.std(axis=0) + 1e-6

        # 4. Action selection: return the mean of elites 
        return mu