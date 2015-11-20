
from multiprocessing import current_process
from simple_mapreduce import MapReduce
import sys


def file_to_words(filename):
    """Read a file and return a sequence of (word, occurances) values.
    """
    output = []

    sys.stderr.write(' '.join(
        [current_process().name, 'reading', filename, '\n']
    ))
    with open(filename, 'rt') as f:
        for line in f:
            for word in line.split():
                word = word.lower()
                output.append((word, 1))
    return output


def count_words(item):
    """Convert the partitioned data for a word to a
    tuple containing the word and the number of occurances.
    """
    word, occurances = item
    return (word, sum(occurances))


if __name__ == '__main__':
    from argparse import ArgumentParser
    import operator
    import glob

    parser = ArgumentParser()
    parser.add_argument('dir')
    args = parser.parse_args()
    in_dir = args.dir

    input_files = glob.glob(in_dir + "/*")

    mapper = MapReduce(file_to_words, count_words)
    word_counts = mapper(input_files)
    word_counts.sort(key=operator.itemgetter(1))
    word_counts.reverse()

    for word, count in word_counts:
        print '%s %d' % (word, count)
