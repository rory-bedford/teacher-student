# Fully Observed

Training a student network to recover perturbed synaptic scaling factors when all neurons are observed. This is the simplest teacher-student setup — no hidden neurons, no EM — serving as a baseline for more complex experiments.

## Method

Synaptic weights are perturbed by cell-type-specific scaling factors drawn from a log-normal distribution. The student receives the teacher's exact spike trains as input and must recover the correct scaling factors (which should converge to 1.0 when normalised by the target).

Scaling factors are optimised via gradient descent (Adam) using Van Rossum loss.

**Scripts:**
- `train.py` — full training with all loss terms (Van Rossum, firing rate, silence penalty)
- `train_single.py` — minimal single-neuron baseline with Van Rossum loss only, useful for verifying the training pipeline

## Usage

```bash
./run experiments/teacher-student/fully-observed/experiment.toml
```

Grid search over seeds for convergence testing:

```bash
./run --grid experiments/teacher-student/fully-observed/experiment.toml
```

## Analysis

`analysis.ipynb` produces:
- Scaling factor trajectories during training
- Loss curves
- Final scaling factor values compared to target
