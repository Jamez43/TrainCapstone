# Summary
This capstone project for The Coding School's Introduction to Artificial Intelligence course explores the challenges of predicting MLB All-Star selections using machine learning. The dataset was sourced from Kaggle’s [Baseball Databank](https://www.kaggle.com/datasets/open-source-sports/baseball-databank), containing player statistics from 2000–2015.

An in depth article about this project is avilable [here](https://medium.com/p/db24c0cb4b20) on Medium.

# Key steps:
- Data preparation: removed irrelevant features, filtered players (≥200 at-bats), and created a new target column ASNextYear.
- Feature engineering: selected core batting stats (AB, R, H, 2B, 3B, HR, RBI, BB, SO).
- Modeling: trained multiple neural networks (baseline, standardized, normalized) with ReLU hidden layers and sigmoid outputs.
- Results: best models achieved ~88% accuracy, showing potential for predicting short-term performance.
- Reflection: highlighted challenges of long-term predictions in baseball and proposed next step of developing a pitch classification model using TrackMan data.
