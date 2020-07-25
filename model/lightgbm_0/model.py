import lightgbm as lgb

def learning(train_X, train_y, test_x, **params):
    dtrain = lgb.Dataset(train_X, label=train_y)
    gbm = lgb.train(params, dtrain)
    preds = gbm.predict(test_x)
    return preds