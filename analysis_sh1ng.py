import pandas as pd
import numpy as np
import lightgbm as lgb
import sys
import gc

## run lightgbm model on one fold of sh1ng data. Take 60% of eligible training
## data randomly for each fold to prevent memory issues
v = int(sys.argv[1])

extratrain_df = pd.read_hdf('data/sh1ng_extratrain.h5', 'table')

extratrain_df = extratrain_df.sample(frac=0.6)

train_df = pd.read_hdf('data/sh1ng_train.h5', 'table')

valid_1 = train_df.query('validation_set == @v')

valid_2 = train_df.query('validation_set == 10')

train_df = train_df.query('validation_set != @v and validation_set != 10').sample(frac=0.6)

train_df = pd.concat([train_df, extratrain_df], ignore_index=True)
del extratrain_df
gc.collect()

test_df = pd.read_hdf('data/sh1ng_test.h5', 'table')

train_df.drop('validation_set', axis=1, inplace=True)
valid_1.drop('validation_set', axis=1, inplace=True)
valid_2.drop('validation_set', axis=1, inplace=True)

features = [
    # 'reordered_dow_ration', 'reordered_dow', 'reordered_dow_size',
    # 'reordered_prev', 'add_to_cart_order_prev', 'order_dow_prev', 'order_hour_of_day_prev',
    'user_product_reordered_ratio', 'reordered_sum',
    'add_to_cart_order_inverted_mean', 'add_to_cart_order_relative_mean',
    'reorder_prob',
    'last', 'prev1', 'prev2', 'median', 'mean',
    'dep_reordered_ratio', 'aisle_reordered_ratio',
    'aisle_products',
    'aisle_reordered',
    'dep_products',
    'dep_reordered',
    'prod_users_unq', 'prod_users_unq_reordered',
    'order_number', 'prod_add_to_card_mean',
    'days_since_prior_order',
    'order_dow', 'order_hour_of_day',
    'reorder_ration',
    'user_orders', 'user_order_starts_at', 'user_mean_days_since_prior',
    # 'user_median_days_since_prior',
    'user_average_basket', 'user_distinct_products', 'user_reorder_ratio', 'user_total_products',
    'prod_orders', 'prod_reorders',
    'up_order_rate', 'up_orders_since_last_order', 'up_order_rate_since_first_order',
    'up_orders', 'up_first_order', 'up_last_order', 'up_mean_cart_position',
    # 'up_median_cart_position',
    'days_since_prior_order_mean',
    # 'days_since_prior_order_median',
    'order_dow_mean',
    # 'order_dow_median',
    'order_hour_of_day_mean',
    # 'order_hour_of_day_median'
]
features.extend([str(s) for s in range(32)])
categories = ['product_id', 'aisle_id', 'department_id']
features.extend(categories)

lgb_train = lgb.Dataset(train_df[features].values, label=train_df.reordered, feature_name=features, categorical_feature=categories)
lgb_valid = lgb.Dataset(valid_1[features].values, label=valid_1.reordered, feature_name=features, categorical_feature=categories)

# specify your configurations as a dict
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'num_leaves': 256,
    'min_sum_hessian_in_leaf': 20,
    'max_depth': 12,
    'learning_rate': 0.05,
    'feature_fraction': 0.6,
    'verbose': 1
}

print('Start training...')
# train
gbm = lgb.train(params,
                lgb_train, valid_sets=[lgb_train, lgb_valid],
                valid_names=['train', 'valid'],
                num_boost_round=1500, early_stopping_rounds=50, verbose_eval=5)

gbm.save_model('sh1ng'+str(v)+".txt")

test_df['prediction'] = gbm.predict(test_df[features].values)
test_df['reordered'] = 0
valid_2['prediction'] = gbm.predict(valid_2[features].values)
valid_2.reordered = 1 * valid_2.reordered
rawpredictions = pd.concat([test_df[['eval_set', 'order_id', 'product_id', 'user_id', 'prediction', 'reordered']],
                           valid_2[['eval_set', 'order_id', 'product_id', 'user_id', 'prediction', 'reordered']]],
                          ignore_index=True)
rawpredictions.to_csv("rawpredictions/sh1ng" + str(v) + ".csv", index=False)
