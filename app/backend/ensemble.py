from time import time
from .feature_based_approach.feature_based import FeatureBased
from .cross_checking_approach.cross_checking_module import CheckText
from .neural_network_approach.neural_network import NeuralNetwork, RNNclassifier
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Ensembling:
    def __init__(self, name='averaging', device=device, nn_ckpt='models/newmodel.pt', nn_w2t='models/word2token.pkl'):
        self.name = name
        self.device = device
        self.feba = FeatureBased()
        self.crch = CheckText()
        self.nene = NeuralNetwork('rnn', self.device)
        self.nene.fit(ckpt=nn_ckpt, w2t=nn_w2t)
        
        
    def predict(self, text):
        st = time()
        pred1 = True if not self.feba.predict(text, params={'load': -0.3, 'rep': 2, 'plur': 2, 'spell': 2, 'punct': 0, 'excess': 0, 'past': 0.0})[0] else False
        f = time()
        print(f'Feature-based predicton made in {round(f-st, 2)}s')
        pred2 = self.crch.check(text)
        c = time()
        print(f'Cross-cheking predicton made in {round(c-f, 2)}s')
        pred3 = True if bool(self.nene.predict(text)) else False
        n = time()
        print(f'NN predicton made in {round(n-c, 2)}s')
        print(pred1, pred2, pred3)
        preds = [pred1, pred2, pred3]
        if preds.count(True) > preds.count(False):
            return True, preds
        return False, preds
