import argparse
import json
import os
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn import metrics
from joblib import dump, load

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-data', help='path to training data file')
    parser.add_argument('--test-data', help='path to test data file')
    parser.add_argument('--output-dir', default='.', help='path to output directory')
    parser.add_argument('--model-dir', help='path to model directory')
    parser.add_argument('--test', action='store_true', help='test mode')
    args = parser.parse_args()
    return args

def read_csv(path, labeled=True):
    data = pd.read_csv(path)
    ids = data['Id']
    X = data[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
    if labeled:
        y = data.Species
    else:
        y = None
    return X, y, ids

def train(train_path, test_path, save_path):
    train_X, train_y, _ = read_csv(train_path)
    test_X, test_y, _ = read_csv(test_path)

    model = svm.SVC()
    model.fit(train_X,train_y)
    prediction = model.predict(test_X)
    acc = metrics.accuracy_score(prediction, test_y)
    print('Accuracy on the dev set: {:.4f}'.format(acc))

    model_path = os.path.join(save_path, 'model.joblib')
    dump(model, model_path)
    print('Model saved to {}'.format(model_path))

def test(test_path, model_path, output_path):
    X, _, ids = read_csv(test_path, labeled=False)
    model = load(os.path.join(model_path, 'model.joblib'))
    prediction = model.predict(X)

    # write results to a JSON file
    results = {i: p for i, p in zip(ids, prediction)}
    output_path = os.path.join(output_path, 'preds.json')
    with open(output_path, 'w') as fout:
        json.dump(results, fout)
    print('Predictions saved to {}'.format(output_path))


def main(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if not args.test:
        train(args.train_data, args.test_data, args.output_dir)
    else:
        test(args.test_data, args.model_dir, args.output_dir)

if __name__ == '__main__':
    args = parse_args()
    main(args)
