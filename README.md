# Simulation-based inference for the Galactic Center Excess

## TODO

- [X] Add more training options (optimization, early stopping)
- [X] More embedding net options
- [ ] Move all DeepSphere code into the `sbi` folder and rename/refactor that folder
- [X] Add hyperparameter options and log them
- [ ] Add a notebook for Poissonian scan to inform parameter ranges
- [ ] Move `utils` into `simulations` folder
- [X] Write complex simulator
    - [X] Output should be unnormalized; make sure Z-scoring takes care of that?
    - [X] Output variance as auxiliary variable? 
- [X] Perform experiments with complex simulator
- [ ] Better treatment of priors (save with simulator)
- [X] More simulator options---wider priors and add Model O etc
- [X] More flexible specification of FC layers
- [ ] Speed up PS simulation
- [ ] Better experiment management (run specific combinations of hyperparameters)