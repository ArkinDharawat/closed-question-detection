import re
from bs4 import BeautifulSoup
import string
"""
Taken from: https://www.kaggle.com/ojwatson/stack-overflow-rudeness-fill-in-blanks#3.-Build-our-data-set
"""
from html.parser import HTMLParser

JOIN_CHAR = ' '
TOKEN_SEP = '|'



class MLStripper(HTMLParser):
    def __init__(self):
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.fed = []

    def handle_data(self, d):
        self.fed.append(d)

    def get_data(self):
        return ''.join(self.fed)


def body_strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()


def remove_filpaths(x):
    return re.sub(r'[[\w\-\:]*\/[\w\-\:]*\/[\w\-\:]*]*|[[\w\-\:\.]*\\[\w\-\:]*\\[\w\-\:]*]*', "", x)


def just_text(x):
    """Method to clear out code-block, p-tags and hyperlinks
    Args:
        x:  html string
    Returns:
        cleaned string
    """
    return re.sub("\n<pre><code>.*?</code></pre>|</p>|<p>|<code>.*?</code>|\n|\t|\r|<a.*>.*?</a>", "", x, flags=re.DOTALL)


def filter_sentence(x):
    filter_regex = '[!"#$%&()*,-./:;<=>?@[\\]^_`{|}~\t\n]'
    filter_char = lambda a: re.sub(filter_regex, "", a, flags=re.VERBOSE)
    filtered_str = [filter_char(x) for x in x.strip().split()]
    filtered_str = [char for char in filtered_str if char != ""]
    return filtered_str


def get_tag_list(text):
    regex = r"<|>"  # "<.*?>.*?<\/.*?>"
    split_list = re.split(regex, text, maxsplit=0, flags=re.IGNORECASE)
    return TOKEN_SEP.join(filter(lambda x: x != '', map(lambda x: x.strip(), split_list)))
