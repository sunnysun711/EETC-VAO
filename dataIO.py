# -*- coding: utf-8 -*-
# read data from original CZ railway directories
import json
import os
import time
from datetime import datetime as dt

import numpy as np
import pandas as pd

CZ_header = {'bj': ['序号', '标记所在位置起始公里标', '标记类型', '标记数据', '标记所在位置的线路公里标增减方向',
                    '标记所在位置对应的线路号'],
             'dl': ['公里标记', '起点公里', '终点公里', '公里标增减方向', '线路号'],
             'pd': ['序号', '起点公里', '坡度', '坡长', '公里标增减方向', '线路标高', '线路号', '公里标记'],
             'qx': ['序号', '曲线起点位置公里标', '曲线半径', '曲线长', '曲线方向', '曲线起点所在位置公里标增减方向',
                    '曲线起点所在位置对应的线路号']
             }
ROOT = os.path.abspath('.')


def execution_timer(func):
    """
    装饰器
    :param func:
    :return:
    """

    def wrapper(*args, **kw):
        a = time.time()
        start_str = dt.now().strftime(format="%m-%d %H:%M:%S")
        print(f"[INFO] {func.__name__}() executing now! ({start_str})")

        res = func(*args, **kw)  # 函数本体

        b = time.time()
        finish_str = dt.now().strftime(format="%m-%d %H:%M:%S")
        print(f"[INFO] {func.__name__}() executed in {b - a:.5f} seconds, finished {finish_str}, started {start_str}")
        return res  # must return the result of the function, otherwise, print(func()) will be None.

    return wrapper


def print_decorated(title: str, *args):
    """

    :param title: string, to be displayed on the start and end lines
    :param args: variables to be displayed in the main body
    :return:
    """
    str_length = 150
    top_ruler = title.center(str_length, '=')
    bottom_ruler = title.center(str_length, '=')
    print(top_ruler)
    for a in args:
        print(a)
    print(bottom_ruler)
    return


def get_random_seed(title='', display=True):
    seed = np.random.randint(10000000)
    if display:
        print_decorated(f'RANDOM-{title}', f"numpy random seed is {seed}.")
    return seed


def read_data(*args: str) -> pd.DataFrame | dict | list:
    _l: list[str] = [i for i in args]
    _l.insert(0, ROOT)
    full_path: str = str(os.path.join(*_l))
    file_format: str = _l[-1].split(".")[-1]
    if file_format == 'csv':
        try:
            data: pd.DataFrame = pd.read_csv(full_path)
        except UnicodeDecodeError:
            data: pd.DataFrame = pd.read_csv(full_path, encoding="gbk")
    elif file_format == 'pkl':
        data: pd.DataFrame = pd.read_pickle(full_path)
    elif file_format == 'tsv':
        try:
            data: pd.DataFrame = pd.read_csv(full_path, sep='\t')
        except UnicodeDecodeError:
            data: pd.DataFrame = pd.read_csv(full_path, sep='\t', encoding="gbk")
    elif file_format == 'json':
        with open(full_path, 'r', encoding="utf-8") as f:
            data: dict | list = json.load(f)
    else:
        raise Exception()
    return data


def main():
    pass


if __name__ == '__main__':
    main()
