import re
import unicodedata

BLANK = ' '

RE_OPS = re.I | re.M | re.S
RE_USR = re.compile(r"""@\S+""", RE_OPS)
RE_TAG = re.compile(r"""#\S+""", RE_OPS)
RE_URL = re.compile(r"""(http|ftp|https)://\S+""", RE_OPS)
RE_NUM = re.compile(r"""[-+]?\d+\.?\d*""", RE_OPS)

SYMBOLS_ = "()[]¿?¡!{}~<>|"
SYMBOLS = set(";:,.@\\-\"/" + SYMBOLS_)

def norm_chars(text):
    L = []
  
    for u in unicodedata.normalize('NFD', text):
        o = ord(u)
        if 0x300 <= o and o <= 0x036F:
            continue
           
        if u in ('\n', '\r', BLANK, '\t'):
            if len(L) == 0:
                continue

            u = BLANK
        
        if u in SYMBOLS:
            if len(L) > 0 and L[-1] != BLANK:
                L.append(BLANK)
            
            L.append(u)
            L.append(BLANK)
            continue
        
        L.append(u)

    return "".join(L)


def preprocess(text):
    text = RE_URL.sub("_url ", text)
    text = RE_USR.sub("_usr ", text)
    text = RE_TAG.sub("_htag ", text)
    text = RE_NUM.sub("0 ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"&lt;", "<", text)
    text = norm_chars(text.lower())
    text = re.sub(r"j(a|e|i)[jaei]+", r"j\1j\1", text)
    text = re.sub(r"h(a|e|i)[haei]+", r"j\1j\1", text)
    return re.sub(r"\s+", BLANK, text)