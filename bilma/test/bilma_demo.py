from bilma import bilma_model

class bilma_demo():
    def __init__(self, model_file):
        self.model, self.tokenizer = self.load_model(model_file)
    
    def pred(self, text, bs=10):
        n = len(text)
        emo = []
        st = 0
        for st in range(0, n, bs):
            toks = self.tokenizer.tokenize(text[st: st + bs])
            p = self.model.predict(toks)
            emo += self.tokenizer.decode_emo(p[1])
        
        return emo
    

    def load_model(self, model_file):
        model = bilma_model.load(model_file)
        tokenizer = bilma_model.tokenizer(vocab_file="bilma/resources/vocab_file_ALL.txt", max_length=280)
        return model, tokenizer
    
    def emoticons(self):
        return self.tokenizer.emo_labels