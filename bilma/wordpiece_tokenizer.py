#from joblib import Parallel, delayed

class Tokenizer():
    def __init__(self, vocab_file, unk_token="[UNK]", end_token="[END]"):
        self.word2idx = {}
        self.idx2word = []
        c = 0
        with open(vocab_file, "r", encoding="utf8") as f:
            while True:
                line = f.readline()
                if not line:
                    break
                self.word2idx[line[0:-1]] = c
                self.idx2word.append(line[0:-1])
                c += 1
        self.n_jobs = 2
        self.UNK = unk_token
        self.END = end_token
        
    def split(self, s):
        split = []
        i = 0
        while i < len(s):
            for j in range(i, len(s)):
                if (s[j] in "abcdefghijklmnopqrstuvwxyz0123456789"):
                    continue
                if (j==i):
                    if (s[j] != " "):
                        split.append(s[i:j+1])
                    i = j + 1
                    break
                split.append(s[i:j])
                i = j
                break
            else:
                split.append(s[i:j+1])
                i=j+1
        return split
    
    def tokenize(self, S):        
        #return Parallel(n_jobs=self.n_jobs)(delayed(self._tokenize)(s) for s in S)
        return [self._tokenize(s) for s in S]
    
    def detokenize(self, S, human_readable=True):        
        #return Parallel(n_jobs=self.n_jobs)(delayed(self._detokenize)(s) for s in S)
        return [self._detokenize(s, human_readable=human_readable) for s in S]
    
    def _tokenize(self, s):
        tokens = []
        s = s.rstrip('\n')
        for w in self.split(s):
            if w in self.word2idx:
                tokens.append(self.word2idx[w])
            else:
                if (len(w)==1):
                    tokens.append(self.word2idx[self.UNK])
                    continue
                                                
                subtoken = []
                l = 0
                while len(w)>l:
                    l = 2
                    for i in range(len(w),0,-1):
                        if (w[0: i] in self.word2idx):
                            subtoken.append(self.word2idx[w[0: i]])
                            break
                    if (i == 0):
                        subtoken = [self.word2idx[self.UNK]]                
                        break
                    w = "##" + w[i: ]
                tokens += subtoken
        return tokens
        
    def _detokenize(self, tokens, human_readable=True):
        sentence = []
        start = 0 if human_readable == False else 1
        
        for t in tokens[start:]:
            c = self.idx2word[t]
            if (human_readable and c == self.END):
                break
            sentence.append(c)
        return sentence