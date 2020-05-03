#!/user/bin/python3
import pandas as pd
import matplotlib.pyplot as plt


def toDf(item_path, dropCols=None):
    pf_list = []
    counter = 0
    for chunk in chunks(item_path):
        if dropCols is not None:
            chunk = chunk.drop(labels=dropCols, axis='columns')
        pf_list.append(chunk)
        counter += len(chunk.index)
        print('read %d of %s' % (counter, item_path))
    return pd.concat(pf_list)


def chunks(item_path):
    chunkSize = 1000000
    return pd.read_csv(item_path, header=0, chunksize=chunkSize)
