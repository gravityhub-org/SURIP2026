import pandas as pd

def h5store(filename, df):
    with pd.HDFStore(filename) as store:
        store.put('df', df)
        store.get_storer('df').attrs.metadata = df.attrs

def h5load(filename):
    with pd.HDFStore(filename) as store:
        dataframe = store['df']
        dataframe.attrs = store.get_storer('df').attrs.metadata
    return dataframe