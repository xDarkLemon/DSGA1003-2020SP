The goal of this practice is to get familiar with [Codalab](https://github.com/codalab/codalab-worksheets) and its competition submission feature,
which has been used for notable benchmarks such as [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/).
We use the [Iris dataset](https://archive.ics.uci.edu/ml/datasets/Iris), which is quite easy as ML techniques is not the focus here.

### Environment
- `python3.6`
- `pip install -r requirements.txt`

### Train your model locally
Let's first make sure that we can run the files locally.

1. Train with data in `data/iris` and save results to `./output`.
```
python main.py --train-data data/iris/train.csv --test-data data/iris/dev.csv --output-dir output
```
2. Test on the dev data using the saved model and save predictions to a JSON file. 
```
python main.py --test-data data/iris/dev.csv --test --model-dir output --output-dir .
```
3. Score the results. You should get an accuracy of 95%.
```
python scorer.py --test-data data/iris/dev.csv --pred-json preds.json
```

### Run on Codalab
#### Create the docker environment
Codalab uses docker to [manage environments](https://github.com/codalab/codalab-worksheets-old/wiki/Creating-Docker-Images).
Let's build the example `Dockerfile`.
```
docker build -t [Docker Hub id]/dsga1003:latest .
```
Once build is done, you should be able to see the image by running
```
docker images
```
Let's push it to your Docker Hub repo.
```
docker tag [Docker Hub id]/dsga1003 [Docker Hub id]/dsga1003:latest
```
You might need to specify `--docker-image` when running your code on Codalab.

#### Submit on Codalab
Please refer the [submission tutorial](https://worksheets.codalab.org/worksheets/0x4405ecda88c94e51a978d921986dc1be).
You can learn more about Codalab [here](https://codalab-worksheets.readthedocs.io/en/latest/).
