"""
Planner selects the next best action to take given the latent embedding of
predicted observations
"""
import numpy as np
import torch
import torch.nn.functional as F

class Planner:
    def __init__(self, max_iter, n_samples, n_elites, planning_horizon, action_dim):
        self.max_iter = max_iter
        self.n_samples = n_samples
        self.n_elites = n_elites
        self.horizon = planning_horizon
        self.action_dim = action_dim
    
    def objective_function(self, z_curr, z_g):
        """
        Computes the squared difference between the predicted final latent embedding 
        at the end of a time horizon (z_H) with the latent embedding of the target (z_g)
        to evaluate if the model reached the goal state
        """
        return F.mse_loss(z_curr.reshape(-1), z_g.reshape(-1))

    def planner(self, lewm, obs, obs_goal, cam_mean, cam_std, sigreg_lambd=0.1, warm_start=None):
        """
        Samples and selects the best action sequence to take given a single
        current observation and goal observation.
        - Cross Entropy Method (CEM): selects 300 candidate action sequences up to time horizon (H)
        - LeWM rollout: actions are applied to current observation to predict the next observation embedding
        - Action selection: the best action seq selected based on the lowest planning cost
        
        Input: current observation and goal observation frame
        Outputs: (H, A) best action sequence found after max_iter
        """
        # parameters are used for sampling distributions
        if warm_start is None:
            # initialize mu and sigma for camera sampling distribution
            mu = np.zeros((self.horizon, 2))
            sigma = np.ones((self.horizon, 2))
            # initialize probability for binary actions sampling
            p = np.ones((self.horizon, 8)) * 0.5
        else:
            mu = warm_start["mu"]
            sigma = warm_start["sigma"]
            p = warm_start["p"]
        # disable camera control
        mu[:] = 0.0
        sigma[:] = 0.0
        print(f"planner parameters: mu={mu.shape}, sigma={sigma.shape}, p={p.shape}")

        # send to gpu if available
        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        obs = obs.to(device)
        obs_goal = obs_goal.to(device)
        lewm.to(device)

        # encode the obs and obs_goal
        with torch.no_grad():
            lewm.eval()
            z1 = lewm.encoder(obs) # current frame
            zg = lewm.encoder(obs_goal) # ex: tree
        current_goal_mse = self.objective_function(z1, zg).item()
        print(f"planner: encoded obs {tuple(z1.shape)}, goal {tuple(zg.shape)}")

        best_z_pred_hist = None # store best prediction embedding
        best_score = None
        for cem_iter in range(self.max_iter):
            print(
                f"Planner iteration: {cem_iter + 1}/{self.max_iter}: "
                f"rolling out {self.n_samples} samples (H={self.horizon})"
            )

            # 1. Action sampling: sample 300 candidate action samples (n_samples)
            #   each sample contains action sequences up to time horizon (H)
            samples = np.zeros((self.n_samples, self.horizon, self.action_dim))
            # Bernoulli sampling for binary actions
            samples[..., :8] = np.random.binomial(1, p=p, size=(self.n_samples, self.horizon, 8))
            # Gaussian sampling for camera actions
            # samples[..., 8:] = np.random.normal(mu, sigma, size=(self.n_samples, self.horizon, 2))
            samples[..., 8:] = 0.0

            scores = []
            rollout_hists = []
            for sample_idx, actions in enumerate(samples):
                # 2. Rollout actions in world model (imagination)
                # store history of past 8 observations and actions. recall that predictor
                # only has access to last 8 actions/obs in its memory
                z_pred_hist = [z1.view(1, 1, -1)]
                a_emb_hist = []
                
                for t in range(self.horizon):
                    # embed and store each sampled action in the imagination horizon
                    a_t = torch.as_tensor(actions[t], dtype=torch.float32, device=device).view(1, 1, -1)
                    # a_t = normalize_camera(a_t, cam_mean, cam_std)
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

                # final predicted embedding at end of horizon (192,)
                z_H = z_pred_hist[-1].view(-1)

                # 3. Compute cost: how close imagined final state is to fixed goal zg
                # add current score
                score = self.objective_function(z_H, zg).item()
                scores.append(score)
                rollout_hists.append(z_pred_hist)

                if (sample_idx + 1) % 50 == 0 or sample_idx + 1 == self.n_samples:
                    print(
                        f"Planner iteration: {cem_iter + 1}/{self.max_iter}: "
                        f"{sample_idx + 1}/{self.n_samples} rollouts scored"
                    )

            #  update camera distribution parameters based on elites
            elite_idx = np.argsort(scores)[:self.n_elites]
            elites = samples[elite_idx]
            top5 = [scores[i] for i in elite_idx[:5]]
            print(f"top 5 best scores: {', '.join(f'{s:.4f}' for s in top5)}")
            mu = elites[..., 8:].mean(axis=0)
            sigma = elites[..., 8:].std(axis=0) + 1e-6
            # update action probabilities based on elites
            p = elites[..., :8].mean(axis=0)
            
            # get best elite score for metrics
            best_elite_idx = elite_idx[0]
            best_score = scores[best_elite_idx]
            best_z_pred_hist = rollout_hists[best_elite_idx]
            print(f"Planner iteration: {cem_iter + 1}/{self.max_iter}: {best_score:.4f}")

        # 4. Action selection: return the mean of elites
        z_H = best_z_pred_hist[-1]
        final_goal_mse = self.objective_function(z_H, zg).item()
        # imagined_latents = torch.cat(best_z_pred_hist[1:], dim=1).squeeze(0)

        planning_losses = {
            "current_goal_mse": current_goal_mse, # current diff with goal
            "cem_best_cost": best_score,
            "final_goal_mse": final_goal_mse, # final diff at H with goal
            "current_latent": z1.detach(),
            "goal_latent": zg.detach(),
            # "imagined_latents": imagined_latents.detach(),
        }
        distribution_params = {
            "mu": mu,
            "sigma": sigma,
            "p": p,
        }
        # print(
        #     f"FINAL SCORE ======"
        #     f"current_goal_mse={planning_losses['current_goal_mse']:.4f},"
        #     f"cem_best_cost={planning_losses['cem_best_cost']:.4f}, "
        #     f"final_goal_mse={planning_losses['final_goal_mse']:.4f}"
        # )
        
        # return best action sequence
        return samples[best_elite_idx], planning_losses, distribution_params
