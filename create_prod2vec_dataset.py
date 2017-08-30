import pandas as pd
import numpy as np
import os
import sys

def create_list(df):
    add_to_cart_order = df.add_to_cart_order.values
    values = df.product_id.values
    index = np.argsort(add_to_cart_order)
    values = values[index].tolist()
    return values



if __name__ == '__main__':

    is_extra = len(sys.argv) == 2 and sys.argv[1] == "extra"
    if is_extra:
        prior_name = "order_products__prior_extratrain.csv"
        train_name = "order_products__train_extratrain.csv"
    else:
        prior_name = "order_products__prior.csv"
        train_name = "order_products__train.csv"
    path = 'data'
    order_prior = pd.read_csv(os.path.join(path, prior_name), dtype={'order_id': np.uint32,
                                                                                      'product_id': np.uint16,
                                                                                      'add_to_cart_order': np.uint8})
    orders = pd.read_csv(os.path.join(path, "orders.csv"), dtype={'order_id': np.uint32,
                                                                  'order_dow': np.uint8,
                                                                  'order_hour_of_day': np.uint8
                                                                  })

    data = pd.merge(order_prior, orders, on='order_id')

    data = order_prior.sort_values(['order_id']).groupby('order_id')['product_id']\
        .apply(lambda x: x.tolist()).to_frame('products').reset_index()
    data = pd.merge(data, orders, on='order_id')
    data.to_pickle(os.path.join(path, 'prod2vec.pkl'))
