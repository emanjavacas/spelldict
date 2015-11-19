import string
import re


punct = re.escape(''.join(c for c in string.punctuation if c != '@'))
subs = re.compile(
    '(%s)' % '|'.join(
        ['\|',                  # remove line breaks
         '\[[^\]]+\]',          # remove [...]
         '^[' + punct + ']+',    # remove leading punctuation
         '[' + punct + ']+$']))  # remove trailing punctuation
process_tokens = [
    lambda token: token if '@' not in token else '',
    lambda token: 'NUM' if re.match(r'.*\d+.*', token) else token,
    lambda token: subs.sub('', token)]


def process_text(text):
    text = text.lower().split()
    sent = []
    for w in text:
        for fn in process_tokens:
            w = fn(w)
        if w:
            sent.append(w)
    return sent
