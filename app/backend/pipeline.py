import argparse


from .neural_network_approach.neural_network import RNNclassifier


def start(article_url):
    print('Backend pipeline started\nURL:',article_url)
    from .ensemble import Ensembling
    from .parse_url import parse_link
    
    text = parse_link(article_url)

    predictor = Ensembling()
    prediction = predictor.predict(text)
    print(prediction)
    
    DummyPercentage = 3.14

    results = dict(likelihood=DummyPercentage)
    results = dict(likelihood=round((1-prediction[1].count(True)/3), 2)*100)
    print(results)
    return results

def main(article_url: str):
    return start(article_url)


    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply fake news detector to an article by providing its url")
    parser.add_argument("--article_url", type=str, help="link to the article")
    args = parser.parse_args()
    main(args.article_url)