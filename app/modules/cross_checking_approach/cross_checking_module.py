from statistics import mean
import torch
from GoogleNews import GoogleNews
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel, PegasusForConditionalGeneration, PegasusTokenizer, AlbertTokenizer, AlbertModel


class CrossChecking:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    def texts_similarity(self, text1, text2):
        encoded_input1 = self.tokenizer(text1, return_tensors='pt')
        output1 = self.model(**encoded_input1)
        encoded_input2 = self.tokenizer(text2, return_tensors='pt')
        output2 = self.model(**encoded_input2)
        result = cosine_similarity(output1['pooler_output'].detach().numpy(), output2['pooler_output'].detach().numpy())
        return result[0][0]
    
    def news_similarity(self, original, texts):
        probs = []
        for text in texts:
            prob = self.texts_similarity(original, text)
            probs.append(prob)
        if len(probs) > 0:
            return mean(probs)
        return 0


class NewsRetrieval:
    def __init__(self, period=None):
        if period:
            self.googlenews = GoogleNews(period=period)
        else:
            self.googlenews = GoogleNews()

    def retrieve(self, summary):
        self.googlenews.get_news(summary)
        output = self.googlenews.get_texts()
        self.googlenews.clear()
        return output


class Summarizer:
    def __init__(self, length=32, model_name='textrank'):
        self.length = length
        if model_name == 'textrank':
            self.model_name = model_name
        else:
            self.model_name = model_name
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.tokenizer = PegasusTokenizer.from_pretrained(self.model_name)
            self.model = PegasusForConditionalGeneration.from_pretrained(self.model_name).to(self.device)

    def summarize(self, text):
        if self.model_name == 'textrank':
            output = summarize(text, words=self.length)
        else:
            batch = self.tokenizer([text], truncation=True, padding='longest', return_tensors="pt").to(self.device)
            translated = self.model.generate(**batch)
            output = self.tokenizer.batch_decode(translated, skip_special_tokens=True)
            output = output[0]

        return output
    
class CheckText:
    def __init__(self, summarizer='textrank', threshold=0.96, emb_model='albert-large-v2'):
        self.summarizer = summarizer
        self.threshold = threshold
        self.summarizer = Summarizer(length=500, model_name=self.summarizer)
        tokenizer = AlbertTokenizer.from_pretrained(emb_model)
        model = AlbertModel.from_pretrained(emb_model)
        self.crosschecking = CrossChecking(tokenizer, model)
        self.newsretrieval = NewsRetrieval()
        
    def check(self, text):
        if len(text.split()) > 512:
            text = self.summarizer.summarize(text)
        texts = self.newsretrieval.retrieve(text)
        score = self.crosschecking.news_similarity(text, texts)
        if score > self.threshold:
            return True
        return False