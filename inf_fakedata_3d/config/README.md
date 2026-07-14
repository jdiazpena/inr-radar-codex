# Configuration

This folder is reserved for explicit, reviewable experiment configuration.

At present, many defaults still live in Python constants and shell arguments. No
configuration file here is silently consumed by a workflow. During the next code
refactoring stage, stable scientific settings should move here in small documented
files, while machine-specific paths and one-off overrides should remain command-line
arguments.

Candidate configurations include:

- benchmark case matrices;
- training and diagnostic cadence;
- regularization targets and lambda bounds;
- dense reconstruction grid settings;
- real-radar window definitions.
