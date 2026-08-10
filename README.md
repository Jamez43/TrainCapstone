# MLB All-Star Predictor

This capstone project investigates whether a player's batting statistics from one season can predict an MLB All-Star selection in the following season. It prepares historical Baseball Databank records, creates a next-season target, and compares neural networks trained on unscaled, standardized, and normalized features.

The experiment's most important result is that accuracy alone is misleading for this dataset. The strongest model recorded approximately **95.15% test accuracy**, but a classifier that always predicts “not an All-Star” would already achieve approximately **95.11% accuracy** because only 238 of 4,866 examples are positive.

An [in-depth article about the project](https://medium.com/p/db24c0cb4b20) is also available on Medium.

## Research Question

Can a player's offensive statistics in one season predict whether that player will be selected as an All-Star in the next season?

The notebook treats this as a binary-classification problem:

```text
ASNextYear = 1  selected as an All-Star in the following season
ASNextYear = 0  not selected as an All-Star in the following season
```

## Dataset

The source data comes from Kaggle's [Baseball Databank](https://www.kaggle.com/datasets/open-source-sports/baseball-databank).

The notebook:

- Uses batting records from 1973 through 2014 when constructing labels
- Matches each batting season with All-Star records from the following year
- Filters out players with fewer than 200 at-bats
- Removes rows with missing values
- Restricts the modeling dataset to batting seasons from 2000 onward
- Excludes pitchers through the available All-Star position field

The final modeling dataset contains:

| Target | Examples | Share |
| --- | ---: | ---: |
| Not an All-Star next season | 4,628 | 95.11% |
| All-Star next season | 238 | 4.89% |
| Total | 4,866 | 100% |

## Features

Nine offensive statistics are used as model inputs:

| Feature | Meaning |
| --- | --- |
| `AB` | At-bats |
| `R` | Runs |
| `H` | Hits |
| `2B` | Doubles |
| `3B` | Triples |
| `HR` | Home runs |
| `RBI` | Runs batted in |
| `BB` | Walks |
| `SO` | Strikeouts |

## Model Architecture

All three experiments use the same Keras network:

```text
9 inputs → 256-unit ReLU layer → 128-unit ReLU layer → 1 sigmoid output
```

Training configuration:

- Adam optimizer with a `0.001` learning rate
- Binary cross-entropy loss
- Batch size of 10
- Three epochs
- Random 50/50 train-test split with `random_state=42`

## Experiments

The notebook compares three versions of the input data:

1. **Unscaled:** original numeric feature values
2. **Standardized:** zero-mean, unit-variance features using `StandardScaler`
3. **Normalized:** features mapped with `MinMaxScaler`

The scalers are fitted only on the training data and then applied to the test data.

## Recorded Results

These values come from the outputs currently saved in `capstone.ipynb`:

| Model | Training accuracy | Test accuracy |
| --- | ---: | ---: |
| Unscaled | 94.12% | 93.18% |
| Standardized | 95.15% | 95.07% |
| Normalized | 95.27% | 95.15% |
| Majority-class baseline | — | 95.11% |

The normalized model produced the highest recorded test accuracy, but its improvement over the majority-class baseline is only about 0.04 percentage points. The notebook does not currently report precision, recall, F1 score, ROC-AUC, or a confusion matrix, so the results do not establish that the model reliably identifies future All-Stars.

## Interpretation

The project demonstrates why a high accuracy value is not sufficient for an imbalanced classification problem. With fewer than 5% positive examples, a model can appear highly accurate while identifying few or no future All-Stars.

A stronger evaluation should prioritize:

- Precision and recall for the All-Star class
- F1 score
- Confusion matrix
- Precision-recall curve
- Comparison with weighted or resampled training
- Evaluation on later seasons that were not used during training

## Run Locally

### Requirements

- Python 3
- Jupyter Notebook or JupyterLab
- NumPy
- Pandas
- TensorFlow / Keras
- Scikit-learn

### Setup

```bash
git clone https://github.com/Jamez43/TrainCapstone.git
cd TrainCapstone
python -m venv .venv
source .venv/bin/activate
pip install jupyter numpy pandas tensorflow scikit-learn
jupyter notebook capstone.ipynb
```

On Windows, activate the environment with:

```powershell
.venv\Scripts\activate
```

Run the notebook from top to bottom. `Batting.csv` and `AllstarFull.csv` are the source files; the notebook generates `CapstoneData.csv` after constructing the next-season target.

## Repository Structure

```text
├── capstone.ipynb     Data preparation, modeling, evaluation, and examples
├── Batting.csv        Historical batting records
├── AllstarFull.csv    Historical All-Star records
├── CapstoneData.csv   Generated dataset with next-season labels
└── README.md          Project methodology and results
```

## Limitations

- The target is highly imbalanced.
- The random split does not simulate predicting genuinely future seasons as well as a chronological split would.
- Only batting statistics are considered; pitching and defensive performance are excluded.
- Selection is influenced by position, league, voting, reputation, team context, injuries, and other factors absent from the model.
- The project evaluates accuracy but does not yet measure positive-class performance.
- The experiment uses historical data through the mid-2010s and may not reflect current selection patterns.

## Next Steps

- Replace the random split with a chronological train-validation-test split
- Add precision, recall, F1, confusion-matrix, and precision-recall metrics
- Compare class weighting, oversampling, and undersampling
- Establish logistic-regression and tree-based baselines
- Tune the decision threshold rather than assuming `0.5`
- Add position, league, age, prior selections, and rate statistics
