import argparse
import json
from sklearn import metrics
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-data', help='path to test data file')
    parser.add_argument('--pred-json', help='JSON file containing the predictions')
    args = parser.parse_args()
    return args

def main(args):
    data = pd.read_csv(args.test_data)
    labels = data.Species
    ids = data.Id
    with open(args.pred_json) as fin:
        results = json.load(fin)

    # make sure the predictions and the groundtruth have the same order
    preds = []
    for i in ids:
        preds.append(results.get(str(i), 0))
    acc = metrics.accuracy_score(labels, preds)
    print('Accuracy on the test set: {:.4f}'.format(acc))

    with open('results.json', 'w') as fout:
        json.dump({'acc': acc}, fout)


if __name__ == '__main__':
    args = parse_args()
    main(args)
