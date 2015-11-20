import sys
import codecs
import string
import re


ROMAN = r'^(?=[mdclxvi])m*(c[md]|d?c*)(x[cl]|l?x*)(i[xv]|v?i*)$'
HYPHEN = r'.*[\-].*'            # multiple repetitions of hyphens
NUM = 'N'                       # ""?
PUNCT = re.escape(string.punctuation)
subs = re.compile(
    '(%s)' % '|'.join(
        ['\|',                # remove line breaks
         '\[[^\]]+\]',        # remove [...]
         '^[%s]+' % PUNCT,    # remove leading punctuation
         '[%s]+$' % PUNCT]))  # remove trailing punctuation
process_tokens = [
    # discard corrupted words
    lambda token: token if '@' not in token else '',
    # normalize numbers
    lambda token: NUM if re.match(r'.*\d+.*', token) else token,
    # and roman numerals
    lambda token: NUM if re.match(ROMAN, token) else token,
    # hyphenised tokens seem problematic
    lambda token: '' if re.match(HYPHEN, token) else token,
    # substitutions
    lambda token: subs.sub('', token),
    # remove single letter tokens except a
    lambda token: '' if len(token.strip()) == 1 and token != 'a' else token]


def process_sent(sent):
    sent = sent.lower().split()
    out = []
    for w in sent:
        for fn in process_tokens:
            w = fn(w)
        if w and len(w) == 1 and w != 'a':
            sys.stderr.write("GOT [%s]\n" % w)
        if w:
            out.append(w)
    return out


if __name__ == '__main__':
    reader = codecs.getreader('utf8')
    writer = codecs.getwriter('utf8')
    sys.stdin = reader(sys.stdin)
    sys.stdout = writer(sys.stdout)
    for line in sys.stdin:
        sys.stdout.write(" ".join(process_sent(line)) + "\n")
