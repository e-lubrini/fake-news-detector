from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel


class CrossChecking():
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained("bert-base-uncased")

    def texts_similarity(text1, text2):
        encoded_input = tokenizer(text1, return_tensors='pt')
        output1 = model(**encoded_input)
        encoded_input2 = tokenizer(text2, return_tensors='pt')
        output2 = model(**encoded_input2)
        result = cosine_similarity(output['pooler_output'].detach().numpy(), output2['pooler_output'].detach().numpy())
        return result[0][0]
