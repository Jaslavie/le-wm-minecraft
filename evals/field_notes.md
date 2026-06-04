## LeWM Minecraft Field Notes

A major challenge is reconciling continuous and binary action steates with a highly dynamic and noisy environment. LeWM is trained on primarily 3rd person POV's of the scene with continuous actions. TLDR is below.

Known challenges:

- Struggles with interpreting OOD object such as diamond blocks that is not in its training data
- Struggles with combined continuous and binary actions. Camera angles are known to turn sporadically.

Known successful attempts:

- Environment: Use the single-tree superflat environment with a single object. This mirrors common env setups like push T with a single objective and a 3rd person POV
- Actions: Interestingly, the model performed worse with more complex action spaces and better with less actions. This is because [naive search is very weak](https://arxiv.org/html/2604.26182) on low-level action spaces, thus performs very poorly and gets lost in noise (curse of dimensionality).
  - TLDR: Actions [Forward, Back] in Tree navigation was more successful than the fully action trajectory.
- Other configs: 
  - Time horizon: Keeping the time horizon longer generates smoother actions. Currently at 8
  - CEM samples: similar to time horizon, larger samples (300 samples, 30 elites)

## Proposed Architecture

The proposed architecture aims to guide the planner toward the goal by minimizing suprise. This two step approach optimizes the planner different for the 2 stages of tree chopping:

1. **Navigating to tree:** Goal image is of the tree.
2. **Chopping tree:** Goal image evolves based on different stages of tree chopping: active chopping state and post-chopping (block broken from tree).

Transition from 1 to 2 occurs when a success threshold (i.e. MSE score) is met.

## Experiments


| Exp Name   | Exp Description                                                         | Exp Parameters                                                                                     | Exp Result                                                                                                                                                                   |
| ---------- | ----------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Goal image | Test whether the target image is the main bottleneck.                   | Pre-chop, active-chop, post-chop, oak, birch, `goal_frame.pkl`, chopping fixtures.                 | Pre-chop is best for navigation. Active-chop reduces latent MSE but does not finish chopping. Post-chop is too hard as the first goal.                                       |
| Encoder    | Test whether the representation supports the task.                      | Custom ViT, fine-tuned ResNet, frozen ResNet analysis, linear probes.                              | ResNet checkpoint is the most stable path. ViT needs retraining. Probes are in progress to test whether position and viewpoint are linearly available.                       |
| Scene      | Test whether the environment is too far from the training distribution. | Real seed world, superflat single tree, superflat discrimination scene with tree, cactus, diamond. | Single-tree superflat works best. Real seed is noisy. Discrimination adds semantic distractors that the encoder was not trained to separate.                                 |
| Objective  | Test whether the planner score has enough signal.                       | Raw latent MSE, normalized MSE, cosine-style frame ranking.                                        | Raw MSE gives CEM more score spread. Normalized MSE is useful for analysis but made rollout scores flat. Frame ranking does not prove action-sensitive prediction.           |
| Planner    | Test rollout depth and CEM settings.                                    | Horizon 4 and 8, 100 and 300 samples, 10 and 30 elites, full queue and one-action MPC.             | Horizon 8 with 300 samples and 30 elites matches the earlier working run. One-action MPC reduces drift but is slower. Full queue can amplify bad plans.                      |
| Actions    | Test whether action execution matches imagined actions.                 | Full action sampling, projected actions, camera disabled, bounded action ticks.                    | Bounded ticks are required because Malmo commands persist. Projected actions are cleaner but reduced score variation. Disabling real camera helps keep the tree view stable. |
| Metrics    | Test whether success is real or only latent.                            | Fresh latent logging, `distance_to_tree`, `task_success_distance`, saved rollout frames.           | Ground-truth distance is necessary. Latent MSE alone can be misleading. Rollout frames enable offline ranking and inspection.                                                |


## Next Experiments

1. Conduct tests on different positions and angles of the tree (can lewm still reach goal)
2. More rigorous encoder assessment and finetuning. It struggles with OOD objects like diamond blocks and confuses the sky with the tree at times

