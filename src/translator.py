import re


class MultipleReplace():
    def __init__(self, d):
        self.d = d
        self.regex_text = '(%s)' % "|".join(d.keys())
        self.regex = re.compile(self.regex_text)

    def translate(self, text):
        def matcher(substr):
            if substr in self.d:
                return self.d[substr]
            else:
                return ''

        return self.regex.sub(
            lambda m: matcher(m.string[m.start(): m.end()]),
            text
        )

    def __str__(self):
        return self.regex_text
