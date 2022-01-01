from .feature_based_approach.feature_based import FeatureBased
from .cross_checking_approach.cross_checking_module import CheckText
from .neural_network_approach.neural_network import NeuralNetwork, RNNclassifier
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Ensembling:
    def __init__(self, name='averaging', device=device, nn_ckpt='model.pt', nn_w2t='word2token.pkl'):
        self.name = name
        self.feba = FeatureBased()
        self.crch = CheckText()
        self.nene = NeuralNetwork('rnn', device)
        self.nene.fit(ckpt=nn_ckpt, w2t=nn_w2t)
        
        
    def predict(self, text):
        pred1 = True if not self.feba.predict(text)[0] else False
        pred2 = self.crch.check(text)
        pred3 = True if bool(self.nene.predict(text)) else False
        print(pred1, pred2, pred3)
        preds = [pred1, pred2, pred3]
        if preds.count(True) > preds.count(False):
            return True, preds
        return False, preds
