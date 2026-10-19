# Data

The project runs out of the box on a small **synthetic** corpus via
`--demo`, so you need no downloads to try the models.

To train on the real **Stanford Sentiment Treebank**, place the processed
splits here:

```
data/
├── train.parquet
├── dev.parquet
├── test.parquet
└── train_tokens.txt      # one training token per line, for the vocabulary
```

Each parquet row has:

| column | meaning |
| ------ | ------- |
| `tokens` | list of word tokens |
| `ids` | list of vocabulary ids for those tokens |
| `label` | `1` = positive, `0` = negative |
| `is_root` | whether the row is a full sentence (vs. a sub-phrase) |

Because SST is a *treebank*, every sentence also contributes its labelled
sub-phrases. Training on sub-phrases (`is_root = False`) gives the models far
more signal; evaluation is reported on root sentences only.

### GloVe embeddings

Download [GloVe 840B.300d](https://nlp.stanford.edu/projects/glove/) and pass
`--glove path/to/glove.840B.300d.txt`. Only vectors for in-vocabulary words are
loaded; everything else is initialised randomly.
