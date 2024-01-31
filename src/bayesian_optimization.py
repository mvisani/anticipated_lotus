from hyperopt import fmin, tpe, space_eval
from hyperopt.early_stop import no_progress_loss


def hyperopt_optimization(objective, space, max_evals):
    best = fmin(
        fn=objective,  # Objective Function to optimize
        space=space,  # Hyperparameter's Search Space
        algo=tpe.suggest,  # Optimization algorithm (representative TPE)
        max_evals=max_evals,  # Number of optimization attempts
        verbose=True,
        show_progressbar=True,
        early_stop_fn=no_progress_loss(iteration_stop_count=30, percent_increase=0.05),
    )
    best = space_eval(space, best)
    return best
