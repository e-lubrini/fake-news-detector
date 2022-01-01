def start(article_url):
    print('Backend pipeline started\nURL:',article_url)
    from .ensemble import Ensembling
    
    text = 'hello'

    '''
    predictor = Ensembling()
    prediction = predictor.predict(text)
    print(str(prediction))
    '''
    
    DummyPercentage = 3.14

    results = dict(likelihood=DummyPercentage)

    return results