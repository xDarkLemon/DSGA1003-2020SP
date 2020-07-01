train:
	python main.py --train-data data/iris/train.csv --test-data data/iris/dev.csv --output-dir output

test:
	python main.py --test-data data/iris/dev.csv --test --model-dir output

score:
	python scorer.py --test-data data/iris/dev.csv --pred-json output/preds.json
