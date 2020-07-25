import optuna

def create_objective(trial, learning, data, target, metric, validation_size=0.25):
    train_x, validation_x, train_y, validation_y = train_test_split(data, target, test_size=validation_size)

    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'num_leaves': trial.suggest_int('num_leaves', 2, 256),
        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
    }

    preds = learning(train_X, train_y, validation_x, **params)
    score = metric(validation_y, preds)
    return score

