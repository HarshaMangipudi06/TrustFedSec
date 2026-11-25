# Data provenance

The repository ships **no dataset**. The default run uses `make_classification` synthetic data and must never be reported as UNSW-NB15.

For local experiments, the two official split CSVs were fetched from a third-party mirror:

- `https://raw.githubusercontent.com/tichill/UNSW_DATA/master/UNSW_NB15_training-set.csv` — 82,332 rows, SHA-256 `7ec02e7e44d72bd265716b33fda0c7f2188b658e3a4ae1aa2c4b306134cd818c`
- `https://raw.githubusercontent.com/tichill/UNSW_DATA/master/UNSW_NB15_testing-set.csv` — 175,341 rows, SHA-256 `0ebd9eb06d53ae223fbe7948737985e851e00e9c20140508d653de4b1655fdd6`

The filenames in that mirror are the reverse of the official row counts: the file called `training-set` has 82,332 rows and the one called `testing-set` has 175,341. The benchmark does not use those filenames as a split — it concatenates both files, then makes its own stratified 70/15/15 split with the configured seed. The files are kept in `data/raw/`, which is gitignored.

The original source is the [UNSW-NB15 dataset page](https://research.unsw.edu.au/projects/unsw-nb15-dataset). UNSW grants free use for academic research and requires citation of Moustafa and Slay (2015, 2016); commercial use requires permission. This mirror is not the official distributor, so the copy cannot be treated as provenance-verified. Before presenting any result as an UNSW-NB15 benchmark, re-download from the official source, check the row counts and hashes, and cite the papers.

Known loading details: the CSVs carry a UTF-8 BOM on the first header (`\ufeffid`); the loader reads with `utf-8-sig`. `id`, `attack_cat`, IP and port identifiers are dropped before training. `proto`, `service` and `state` are one-hot encoded; numeric columns are median-imputed and standardised. Preprocessing is fit on the training split only.
