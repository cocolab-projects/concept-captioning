# concept-captioning

From concepts to language

## Instructions

Clone the repository. You will need to install several modules including `torch`, `tqdm`, `seaborn`, `dill`, `spacy`, `allennlp`, `torchtext`, `revtok`. (`pip install` should do the trick for most of them, or `conda` if you have it). Basically keep installing things until you stop receiving `ImportError`s.

Once `spacy` (an NLP toolkit) is installed you will also need to run

```bash
python -m spacy download en_core_web_sm
```

to download a small English NLP model for tokenizing, parsing, etc.

## Running the code

The `experiments` folder has subdirectories for models implemented with both raw visual features (`vision`) and binary feature representations (`vectorized`). Within each directory there are different modules for running concept learning (`concept`), reference game (`ref`) and multi-task (`multi`) models. You can ignore the `ref` and `multi` models and focus on the `concept` for now. This is the "student" model we discussed.

```bash
python -m experiments.vectorized.concept
```

runs the student model on binary representations.

It's worth going through the code to see how it works. These scripts import modules from the `data` and `models` folders, so look in there as well.

**important**: when you run on the cluster, change the default `--out-dir` argument to a folder on the `/mnt/fsX` file system where `X` is some number (look at the CCN wiki for why).
