# Match-Ignition
> PyTorch implementation of [CIKM 2021] Match-Ignition: Plugging PageRank into Transformer for Long-form Text Matching.

[![Python 3.6](https://img.shields.io/badge/python-3.6%20%7C%203.7-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellowgreen.svg)](https://opensource.org/licenses/Apache-2.0)

## Usage

### Environment Preparation
```bash
pip install -r requirements.txt
cd transformers
sh switch_version.sh 1
```

### Data Preparation
```bash
cd data
tar xzvf orig.tar.gz
```
Note: the original dataset can be downloaded from [here](https://github.com/BangLiu/ArticlePairMatching).

### Sentence-level Noise Filtering
```bash
python generate_data.py
```

### Word-level Noise Filtering
```bash
python run.py
```

## Citation

If you use Match-Ignition in your research, please use the following BibTex entry.

```
@inproceedings{pang2021matchignition,
    title={Match-Ignition: Plugging PageRank into Transformer for Long-form Text Matching},
    author={Liang Pang and Yanyan Lan and Xueqi Cheng},
    booktitle = {Proceedings of the 30th ACM International Conference on Information and Knowledge Management},
    series = {CIKM'21},
    year = {2021},
}
```

## License

[Apache-2.0](https://opensource.org/licenses/Apache-2.0)

Copyright (c) 2019-present, Liang Pang (pl8787)
