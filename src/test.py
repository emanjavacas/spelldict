import unittest
from utils import build_contexts

text = """Therefore it is thought fitt,\n
to giue notice to all persons in general of his Maiesties.\n
And if any hereafter shall haue occasion of vse any Cloath,\n
and that the buyer and buyers may not incurr his Maiesties,\n
nor bring vpon themselues the paines penalties and imprisonment,\n
be inflicted as aforesaid. Therefore they may repaire to Hunny \n
Cheapside, ouer against Bowe Church, where they shall be dealt\n
withall for the buying of such things ready Printed.\n"""
sents = [s.split(" ") for s in text.split("\n") if s]


class Context5(unittest.TestCase):
    def test(self):
        window = 5
        X, y, w, c = build_contexts(sents, window=window, one_hot_enc=None)
        self.assertEqual(len(X), len(y))
        self.assertEqual(len(X), len([i for s in sents for i in s]))
        self.assertEqual("efore", c.decode_seq(*X[1]))


class Context10(unittest.TestCase):
    def test(self):
        window = 10
        X, y, w, c = build_contexts(sents, window=window, one_hot_enc=None)
        self.assertEqual(len(X), len(y))
        self.assertEqual(len(X), len([i for s in sents for i in s]))
        self.assertEqual("|||to giue", c.decode_seq(*X[7]))


if __name__ == "__main__":
    unittest.main()
