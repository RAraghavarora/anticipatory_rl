# Git setup:

Git sparse only shows files related to blocksworld in this branch.

1) `git sparse-checkout init`
2) `git config --worktree core.sparseCheckoutCone false`
3) Add relevant pattern to `.git/info/sparse-checkout`
ex:
```
*
/anticipatory_rl/
!/paper_restaurant/
!/anticipatory_rl/agents/restaurant*
!/anticipatory_rl/agents/simple_grid*
!/anticipatory_rl/agents/three_box_dqn.py
!/anticipatory_rl/agents/task_conditioned_q_agent.py
!/anticipatory_rl/agents/train_phi.py
!/anticipatory_rl/agents/random_agent.py
```