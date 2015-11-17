# encoding: utf-8
import tarfile
import re
import numpy as np

in_fn = "data/postprocessed.tar.gz"


def partition(lst, n):
    for ix in xrange(0, len(lst), n):
        yield lst[ix:ix+n]


class Indexer(object):
    def __init__(self):
        self.decoder = {}
        self.encoder = {}
        self.last = 0

    def char_indexer(self):
        """
        factory method that creates a char-level indexer
        based on a previously learnt word-level indexer
        """
        indexer = Indexer()
        chars = set([c for w in self.encoder for c in w])
        for c in chars:
            indexer.encode(c)
        return indexer

    def max_word_len(self):
        return max([len(w) for w in self.encoder])

    def to_one_hot(self, n):
        vec = np.zeros(len(self.decoder), dtype=np.int8)
        vec[n] = 1
        return vec

    # def random_vec(self, max_len):

    def _encode(self, s):
        idx = self.encoder.get(s, None)
        if idx is not None:
            return idx
        else:
            idx = self.last
            self.encoder[s] = idx
            self.decoder[idx] = s
            self.last += 1
            return idx

    def encode(self, s, output="int"):
        idx = self._encode(s)
        if output == "int":
            return idx
        if output == "one_hot":
            return self.to_one_hot(idx)

    def decode(self, idx, output="int"):
        if idx not in self.decoder:
            raise ValueError("Cannot found index [%d]" % idx)
        return self.decoder[idx]

process_fns = [
    lambda x: re.sub(r'\[[^\]]+\]', "", x),
    lambda x: x.replace("|", ""),
    lambda x: x.lower(),
    lambda x: x.split(),
    lambda x: [w for w in x if '@' not in w]
]


def process_text(text):
    for fn in process_fns:
        text = fn(text)
    return text


def itertar(in_fn, process_fn=process_text):
    """ generator over lines in tarred """
    with tarfile.open(in_fn, "r|*") as tar:
        for tarinfo in tar:
            doc = tar.extractfile(tarinfo)
            for l in doc:
                processed = process_fn(l)
                if processed:
                    yield processed


def padding(chars, max_len, default):
    if len(chars) == max_len:
        return chars
    if len(chars) > max_len:
        raise ValueError("Sequence longer than max")
    return (max_len - len(chars)) * [default] + chars


def index_sents(sents):
    indexer = Indexer()
    idx_sents = []
    for sent in sents:
        idx_sent = []
        for word in sent:
            idx_sent.append(indexer.encode(word))
        idx_sents.append(idx_sent)
    return idx_sents, indexer


def build_data(sents, word_indexer, char_indexer, default=' '):
    """
    Computes sents to tuples of prev-word, target-word
    transforming the former to one-hot-encoding matrix
    """
    pad_id = char_indexer.encode(' ')
    for sent in sents:
        for i, target in enumerate(sent[1:]):
            prev_idx = sent[i]
            prev_chrs = [c for c in word_indexer.decode(prev_idx)]
            prev_padd = padding(prev_chrs, word_indexer.max_word_len(), pad_id)
            prev_embd = [char_indexer.encode(c, output="one_hot")
                         for c in prev_padd]
            yield (np.asarray(prev_embd), target)


text = """from their godly endeavour, assuring themselves,\n
that though bad Christians carp and repine\n
at the confidence and respect which Catholick\n
Princes and Ministers of State shew to their Clergy,\n
by trusting them with their conscienbces and affaires\n"""


sents = [process_text(sent) for sent in text.split('\n') if sent.split()]
idx_sents, word_indexer = index_sents(sents)
char_indexer = word_indexer.char_indexer()
data = build_data(idx_sents, word_indexer, char_indexer)


def memoize(fn):
    memo = {}

    def wrapper(x):
        in_memo = memo.get(x, None)
        if in_memo:
            print "in Memo!"
            return in_memo
        result = fn(x)
        memo[x] = result
        return result
    return wrapper


@memoize
def add(pair):
    return pair[0] + pair[1]
