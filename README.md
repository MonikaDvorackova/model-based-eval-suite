# Model Based Evaluation Suite

Evaluates a policy in both a learned model and a nonlinear reference environment. Reports predicted return, actual return, their gap, and mean transition error so a planner that exploits model error is visible.

Run `python3 main.py`. Replace the toy dynamics with environment adapters; always report closed-loop environment outcomes, not only prediction loss.
