# create separate files with last prior order used as training order, so this
# data can be used to create extra training data for the sh1ng pipeline
import numpy as np
import pandas as pd

orders_df = pd.read_csv("data/truetrain/orders.csv")
prior_df = pd.read_csv("data/truetrain/order_products__prior.csv")

orders_df = orders_df.query('eval_set == "prior"')
orders_df['max_order'] = orders_df.groupby('user_id').order_number.transform(max)
orders_df.loc[orders_df.order_number == orders_df.max_order, 'eval_set'] = 'train'
orders_df.drop('max_order', axis=1, inplace=True)

train_df = prior_df.loc[prior_df.order_id.isin(orders_df.query('eval_set == "train"').order_id),:]
prior_df = prior_df.loc[prior_df.order_id.isin(orders_df.query('eval_set == "prior"').order_id),:]

orders_df.to_csv('data/orders.csv', index=False)

prior_df.to_csv("data/order_products__prior_extratrain.csv", index=False)
train_df.to_csv("data/order_products__train_extratrain.csv", index=False)
