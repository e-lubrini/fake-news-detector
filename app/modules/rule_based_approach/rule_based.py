from nltk.corpus import stopwords
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
from textblob import Word


class FeatureBased:
    def __init__(self, emotional=True, repetitions=True, plural=True, spelling=True, explanation=True):
        self.explanation = explanation
        self.emotional = emotional
        self.repetitions = repetitions
        self.plural = plural
        self.spelling = spelling
        
        self.nlp = spacy.load('en_core_web_sm')
        self.nlp.add_pipe('spacytextblob')
        self.text = None
        self.prep_doc = None
        self.stopwords = set(stopwords.words('english'))
        
    
    def predict(self, text):
        out = {}
        if self.emotional:
            out['emotional'] = self._emotional_feature(text)
        if self.repetitions:
            out['repetitions'] = self._repetitions_feature(text)
        if self.plural:
            out['plural'] = self._plural_feature(text)
        if self.spelling:
            out['spelling'] = self._spelling_feature(text)
        
        return out
    
    def _emotional_feature(self, text):
        if self.text != text:
            self.text = text
            self.prep_doc = nlp(text)
        score = self.prep_doc._.polarity 
        if self.explanation:
            expl = sum([x[0] for x in self.prep_doc._.assessments], [])
        return (score, expl)
    
    def _repetitions_feature(self, text):
        if self.text != text:
            self.text = text
            self.prep_doc = nlp(text)
        all_words = [token.lemma_ for token in self.prep_doc 
                    if token.pos_ not in ['PUNCT', 'SYM']]
        repeated_words = list(set([word.lower() for word in all_words 
                                   if all_words.count(word) > 1]).difference(self.stopwords))
        return (len(repeated_words), repeated_words)
        
    def _plural_feature(self, text):
        if self.text != text:
            self.text = text
            self.prep_doc = nlp(text)
        plural_forms = []
        for token in self.prep_doc:
            if token.dep_ in ['nsubj', 'ROOT']:
                if token.tag_ == 'PRP':
                    if token.text in ['we', 'they', 'us', 'them']:
                        plural_forms.append(token.text)
                elif token.tag_ == 'NNS':
                    plural_forms.append(token.text)
        return (len(plural_forms), plural_forms)
    
    def _spelling_feature(self, text):
        if self.text != text:
            self.text = text
            self.prep_doc = nlp(text)
        incorrect_words = [token.text for token in self.prep_doc 
                           if Word(token.text).spellcheck()[0][0] != token.text]
        return (len(incorrect_words), incorrect_words)
            
        
    
