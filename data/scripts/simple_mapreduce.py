# inspired by https://pymotw.com/2/multiprocessing/mapreduce.html

from collections import defaultdict
from itertools import chain
from multiprocessing import Pool


class MapReduce(object):
    def __init__(self, map_fn, reduce_fn, num_workers=None):
        self.map_fn = map_fn
        self.reduce_fn = reduce_fn
        self.pool = Pool(num_workers)

    def partition(self, mapped):
        partitioned = defaultdict(list)
        for k, v in mapped:
            partitioned[k].append(v)
        return partitioned.items()

    def __call__(self, inputs, chunksize=1):
        map_responses = self.pool.map(self.map_fn, inputs, chunksize=chunksize)
        partitioned = self.partition(chain(*map_responses))
        reduced = self.pool.map(self.reduce_fn, partitioned)
        return reduced
