# load modules and dataset
"""Utilities for running federated experiments with differential privacy."""

import pandas as pd

from DP_server_private import *
from DP_load_dataset import (
    make_adult_info,
    make_compas_info,
    synthetic_info,
    bank_info,
    bank_z,
    bank_num_features,
)

try:
    from ray.tune.progress_reporter import CLIReporter
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler
    RAY_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    RAY_AVAILABLE = False

def run_dp(method, model, dataset, prn=True, seed=123, epsilon=1, trial=False, dirichlet_alpha=0.1, **kwargs):
    # choose the model
    if model == 'logistic regression':
        arc = logReg
    elif model == 'multilayer perceptron':
        arc = mlp
    else:
        Warning('Does not support this model!')
        exit(1)

    # set up the dataset
    if dataset == 'synthetic':
        Z, num_features, info = 2, 3, synthetic_info
    elif dataset == 'adult':
        info, num_features = make_adult_info(alpha=dirichlet_alpha, seed=seed)
        Z = 2
    elif dataset == 'compas':
        Z, num_features, info = make_compas_info(alpha=dirichlet_alpha, seed=seed)
    elif dataset == 'bank':
        Z, num_features, info = bank_z, bank_num_features, bank_info
    else:
        Warning('Does not support this dataset!')
        exit(1)

    # set up the server
    server = Server(arc(num_features=num_features, num_classes=2, seed = seed), info, train_prn = False, seed = seed, Z = Z, ret = True, prn = prn, trial = trial, epsilon = epsilon)

    # execute
    if method == 'fedfb':
        acc, dpdisp, eod, classifier = server.FedFB(**kwargs)
    elif method == 'fflfb':
        acc, dpdisp, eod, classifier = server.FFLFB(**kwargs)
    else:
        Warning('Does not support this method!')
        exit(1)

    if not trial:
        return {'accuracy': acc, 'DP Disp': dpdisp, 'EOD': eod}

def sim_dp(method, model, dataset, epsilon=1, num_sim=5, seed=0, dirichlet_alpha=0.1, resources_per_trial={'cpu':4}, **kwargs):
    """Hyperparameter tuning with Ray Tune (if available)."""

    if not RAY_AVAILABLE:
        raise ImportError("ray is required for sim_dp but is not installed")
    # choose the model
    if model == 'logistic regression':
        arc = logReg
    elif model == 'multilayer perceptron':
        arc = mlp
    else:
        Warning('Does not support this model!')
        exit(1)

    # set up the dataset
    if dataset == 'synthetic':
        Z, num_features, info = 2, 3, synthetic_info
    elif dataset == 'adult':
        info, num_features = make_adult_info(alpha=dirichlet_alpha, seed=seed)
        Z = 2
    elif dataset == 'compas':
        Z, num_features, info = make_compas_info(alpha=dirichlet_alpha, seed=seed)
    elif dataset == 'bank':
        Z, num_features, info = bank_z, bank_num_features, bank_info
    else:
        Warning('Does not support this dataset!')
        exit(1)

    if method == 'fedfb':
        print('--------------------------------Hyperparameter selection--------------------------------')
        print('--------------------------------Seed:' + str(seed) + '--------------------------------')
        config = {'lr': tune.grid_search([.001, .005, .01]),
                'alpha': tune.grid_search([.001, .05, .08, .1, .2, .5, 1, 2])}

        def trainable(config):
            return run_dp(method=method, model=model, dataset=dataset, prn=False, trial=True,
                          seed=seed, epsilon=epsilon, dirichlet_alpha=dirichlet_alpha,
                          learning_rate=config['lr'], alpha=config['alpha'], **kwargs)

        asha_scheduler = ASHAScheduler(
            time_attr = 'iteration',
            metric = 'disp',
            mode = 'min',
            grace_period = 5)

        reporter = CLIReporter(metric_columns=['loss', 'accuracy', 'iteration', 'disp'])

        analysis = tune.run(
            trainable,
            resources_per_trial = resources_per_trial,
            config = config,
            num_samples = 1,
            scheduler=asha_scheduler,
            progress_reporter=reporter)

        best_trial = analysis.get_best_trial("disp", "min", "last")
        params = best_trial.config
        learning_rate, alpha = params['lr'], params['alpha']

        print('--------------------------------Start Simulations--------------------------------')
        # get test result of the trained model
        server = Server(arc(num_features=num_features, num_classes=2, seed = seed), info, epsilon = epsilon, train_prn = False, seed = seed, Z = Z, ret = True, prn = False)
        trained_model = copy.deepcopy(server.model)
        trained_model.load_state_dict(torch.load(os.path.join(best_trial.checkpoint.value, 'checkpoint')))
        test_acc, n_yz, test_eod = server.test_inference(trained_model)
        df = pd.DataFrame([{'accuracy': test_acc, 'DP Disp': DPDisparity(n_yz), 'EOD': test_eod}])

        # use the same hyperparameters for other seeds
        for seed in range(1, num_sim):
            print('--------------------------------Seed:' + str(seed) + '--------------------------------')
            result = run_dp(
                method=method,
                model=model,
                dataset=dataset,
                prn=False,
                epsilon=epsilon,
                seed=seed,
                dirichlet_alpha=dirichlet_alpha,
                learning_rate=learning_rate,
                alpha=alpha,
                **kwargs,
            )
            df = df.append(pd.DataFrame([result]))
        df = df.reset_index(drop = True)
        acc_mean, dp_mean = df.mean()
        acc_std, dp_std = df.std()
        print("Result across %d simulations: " % num_sim)
        print("| Accuracy: %.4f(%.4f) | DP Disp: %.4f(%.4f)" % (acc_mean, acc_std, dp_mean, dp_std))
        return acc_mean, acc_std, dp_mean, dp_std

    elif method == 'fflfb':
        print('--------------------------------Hyperparameter selection--------------------------------')
        print('--------------------------------Seed:' + str(seed) + '--------------------------------')
        num_clients = len(info[2])
        if num_clients <= 2:
            params_array = cartesian([[.001, .01, .1]]*num_clients).tolist()
            # params_array = cartesian([[.01]]*num_clients).tolist()
            def trainable(config):
                return run_dp(
                    method=method,
                    model=model,
                    dataset=dataset,
                    epsilon=epsilon,
                    prn=False,
                    trial=True,
                    seed=seed,
                    dirichlet_alpha=dirichlet_alpha,
                    learning_rate=0.005,
                    alpha=config['alpha'],
                    **kwargs,
                )
        else:
            params_array = [.001, .002, .005, .01, .02, .05, .1, 1]
            def trainable(config):
                return run_dp(
                    method=method,
                    model=model,
                    dataset=dataset,
                    epsilon=epsilon,
                    prn=False,
                    trial=True,
                    seed=seed,
                    dirichlet_alpha=dirichlet_alpha,
                    learning_rate=0.005,
                    alpha=[config['alpha']] * num_clients,
                    **kwargs,
                )
        config = {'alpha': tune.grid_search(params_array)}

        asha_scheduler = ASHAScheduler(
            time_attr = 'iteration',
            metric = 'disp',
            mode = 'min',
            grace_period = 5)

        reporter = CLIReporter(metric_columns=['loss', 'accuracy', 'iteration', 'disp'])

        analysis = tune.run(
            trainable,
            resources_per_trial = resources_per_trial,
            config = config,
            num_samples = 1,
            scheduler=asha_scheduler,
            progress_reporter=reporter)

        best_trial = analysis.get_best_trial("disp", "min", "last")
        params = best_trial.config
        alpha = params['alpha']

        print('--------------------------------Start Simulations--------------------------------')
        # get test result of the trained model
        server = Server(arc(num_features=num_features, num_classes=2, seed = seed), info, epsilon = epsilon, train_prn = False, seed = seed, Z = Z, ret = True, prn = False)
        trained_model = copy.deepcopy(server.model)
        trained_model.load_state_dict(torch.load(os.path.join(best_trial.checkpoint.value, 'checkpoint')))
        test_acc, n_yz, test_eod = server.test_inference(trained_model)
        df = pd.DataFrame([{'accuracy': test_acc, 'DP Disp': DPDisparity(n_yz), 'EOD': test_eod}])

        # use the same hyperparameters for other seeds
        for seed in range(1, num_sim):
            print('--------------------------------Seed:' + str(seed) + '--------------------------------')
            if num_clients <= 2:
                result = run_dp(
                    method=method,
                    model=model,
                    dataset=dataset,
                    epsilon=epsilon,
                    prn=False,
                    seed=seed,
                    dirichlet_alpha=dirichlet_alpha,
                    learning_rate=0.005,
                    alpha=alpha,
                    **kwargs,
                )
            else:
                result = run_dp(
                    method=method,
                    model=model,
                    dataset=dataset,
                    epsilon=epsilon,
                    prn=False,
                    seed=seed,
                    dirichlet_alpha=dirichlet_alpha,
                    learning_rate=0.005,
                    alpha=[alpha] * num_clients,
                    **kwargs,
                )
            df = df.append(pd.DataFrame([result]))
        df = df.reset_index(drop = True)
        acc_mean, dp_mean = df.mean()
        acc_std, dp_std = df.std()
        print("Result across %d simulations: " % num_sim)
        print("| Accuracy: %.4f(%.4f) | DP Disp: %.4f(%.4f)" % (acc_mean, acc_std, dp_mean, dp_std))
        return acc_mean, acc_std, dp_mean, dp_std
    else: 
        Warning('Does not support this method!')
        exit(1)

def sim_dp_man(method, model, dataset, epsilon=1, num_sim=5, seed=0, dirichlet_alpha=0.1, **kwargs):
    """Run multiple simulations with differential privacy and report statistics."""
    results = []
    for seed in range(num_sim):
        results.append(
            run_dp(
                method,
                model,
                dataset,
                prn=True,
                epsilon=epsilon,
                seed=seed,
                trial=False,
                dirichlet_alpha=dirichlet_alpha,
                **kwargs,
            )
        )

    df = pd.DataFrame(results)

    acc_mean = df['accuracy'].mean()
    acc_std = df['accuracy'].std()
    dp_mean = df['DP Disp'].mean()
    dp_std = df['DP Disp'].std()
    eod_mean = df['EOD'].mean()
    eod_std = df['EOD'].std()

    print("Result across %d simulations: " % num_sim)
    print(
        "| Accuracy: %.4f(%.4f) | DP Disp: %.4f(%.4f) | EOD: %.4f(%.4f)"
        % (acc_mean, acc_std, dp_mean, dp_std, eod_mean, eod_std)
    )

    return acc_mean, acc_std, dp_mean, dp_std, eod_mean, eod_std
