from hyperopt import fmin, tpe


def hyperopt_optimization(objective, space):
    best = fmin(
        fn=objective,  # Objective Function to optimize
        space=space,  # Hyperparameter's Search Space
        algo=tpe.suggest,  # Optimization algorithm (representative TPE)
        max_evals=1000,  # Number of optimization attempts
        verbose=True,
        show_progressbar=True,
    )
    return best
