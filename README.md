# Overview

This project is Team USC's solution to Problem B on SCUDEM 2025, An AI Ouroboros. We try to simulate and plot model collapse for simple gaussians using a polynomial ODE and then extrapolate our results to draw conclusions about the higher dimensional models provided in Shumailov et al. (2024) provided with the problem.

## Project Goal

Our goals included:

Simulate the Collapse: Recursively train a simple Gaussian model on its own output and observe the decay of its Standard Deviation ($\sigma$) (our measure of diversity) over generations.

Model the Phenomenon: Fit the simulation results to the analytical solution of a differential equation, proving that the collapse follows a predictable, non-linear mathematical law.

Answer Key Questions: Use the conclusions to answer how model collapse happens over time (simulated by number of generations) based on the proportion of human and synthetic data it's trained on.

## Simulation Mechanism

The simulation uses an iterative training loop:

Start: The initial model is defined by the True Standard Deviation.

Sample: In each generation, the model draws a finite sample of constant size from its current normal distribution. Multiple trajectories are averaged per generation to get a more accurate result.

Train: The next generation's model is defined by the Standard Deviation of that sample.

Recurse: Because a finite sample is an imperfect estimate, the statistical error causes to be slightly smaller than, leading to compounding, irreversible loss (information cannot be recovered).

##Mathematical Basis (The Link to LLMs)

The collapse in the Gaussian model is a direct mathematical analogue to the collapse of LLMs (tracked by rising perplexity) and VAEs.

The core principle is the Statistical Approximation Error inherent in finite sampling (guaranteed information loss).

1. The Discrete Step (Recurrence Relation):
The variance decreases due to an error term proportional to number of generations and inversely proportional to the sample size.

2. The Continuous Model (Fitted Differential Equation):
We fit the simulation's standard deviation to the solution of a differential equation that accounts for decay to an asymptotic floor.

This decay proves the collapse is non-linear and governed by the current state of diversity, allowing us to extrapolate to higher-dimensional data.

## Key Findings

Non-Linear Decay: The model collapse is rapid initially and slows down as diversity approaches the floor, following the inverse-square root decay predicted by the analytical solution.

Mitigation: The best way to prevent collapse is to introduce some amount of real, non-synthetic data during training of some generation, which forces the stable equilibrium point to move back toward the original standard deviation.

Stable Equilibrium: The system collapses to a stable floor (the asymptote of the fitted curve), rather than truly zero (unless it is completely trained on synthetic data), which is the fixed point of the system when external data is completely absent.

---

We truly learned so much over the course of this project and are very happy to have had the opportunity to compete. A big shout to all members of the team, to SCUDEM USC facilitating this competition and our participation, and most of all to Dr. Paula Vasquez for being such an incredible mentor to us during this process.
