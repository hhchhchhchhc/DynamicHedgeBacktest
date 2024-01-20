#!/usr/bin/env python3
import logging

from utils.async_utils import *
import os
import json
import pandas as pd
import numpy as np
import pyarrow,pyarrow.parquet,s3fs
from datetime import timezone
import retry

# this is for jupyter
import cufflinks as cf
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)

'''
I/O helpers
'''


def to_csv(*args,**kwargs):
    return args[0].to_csv(*args[1:],**kwargs)


async def async_to_csv(*args,**kwargs):
    coro = async_wrap(to_csv)
    return await coro(*args,**kwargs)


def to_parquet(df,filename,mode="w"):
    if mode == 'a' and os.path.isfile(filename):
        previous = from_parquet(filename)
        df = pd.concat([previous,df],axis=0)
        df = df[~df.index.duplicated()].sort_index()
    pq_df = pyarrow.Table.from_pandas(df)
    pyarrow.parquet.write_table(pq_df, filename)
    return None


def from_parquets_s3(filenames,columns=None):
    '''columns = list of columns. All if None
    filename = list'''
    kwargs = {'columns':columns} if columns else dict()
    return pyarrow.parquet.ParquetDataset(filenames,filesystem=s3fs.S3FileSystem()).read_pandas(**kwargs).to_pandas()


def from_parquet(filename):
    result = pyarrow.parquet.read_table(filename).to_pandas()
    result.index = [t.replace(tzinfo=timezone.utc) for t in result.index]
    return result


@retry.retry(exceptions=Exception, tries=3, delay=1,backoff=2)
def ignore_error(func):
    @functools.wraps(func)
    async def wrapper_ignore_error(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logger = logging.getLogger(func.__module__.split('.')[0])
            logger.warning(f'{str(func)}({args}{kwargs})\n-> {e}', exc_info=False)
    return wrapper_ignore_error

'''
misc helpers
'''

import collections
def flatten(dictionary, parent_key=False, separator='.'):
    """
    All credits to https://github.com/ScriptSmith
    Turn a nested dictionary into a flattened dictionary
    :param dictionary: The dictionary to flatten
    :param parent_key: The string to prepend to dictionary's keys
    :param separator: The string used to separate flattened keys
    :return: A flattened dictionary
    """

    items = []
    for key, value in dictionary.items():
        new_key = str(parent_key) + separator + key if parent_key else key
        if isinstance(value, collections.MutableMapping):
            items.extend(flatten(value, new_key, separator).items())
        elif isinstance(value, list):
            for k, v in enumerate(value):
                items.extend(flatten({str(k): v}, new_key).items())
        else:
            items.append((new_key, value))
    return dict(items)

def deepen(dictionary, parent_key=False, separator='.'):
    """
    flatten^-1
    """
    top_keys = set(key.split(separator)[0] for key in dictionary.keys())
    result = {}
    for top_key in top_keys:
        sub_dict={}
        sub_result={}
        for key, value in dictionary.items():
            if key.split(separator)[0]==top_key:
                if separator in key:
                        sub_dict|={key.split(separator, 1)[1]:value}
                else:
                    sub_result |={key:value}
        if sub_dict != {}:
            result |= {top_key:deepen(sub_dict,parent_key=parent_key,separator=separator)}
        else:
            result |= sub_result

    return result

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, pd.core.generic.NDFrame):
            return obj.to_json()
        if isinstance(obj, collections.deque):
            return None
        return super(NpEncoder, self).default(obj)
