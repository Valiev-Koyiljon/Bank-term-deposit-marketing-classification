
# BankX Marketing Campaign Analysis

## Overview
This project explores a marketing dataset from BankX related to direct marketing campaigns (phone calls). The main goal is to predict whether a client will subscribe to a term deposit, thus framing a binary classification problem.

## Dataset
The dataset consists of various attributes ranging from client information to campaign details. The primary task is to predict the binary outcome (`y`) indicating subscription to a term deposit.

### Attributes
- **Client Data:** Age, job, marital status, education, default status, balance, housing loan, personal loan.
- **Campaign Data:** Contact type, last contact day and month, last contact duration, number of contacts in this campaign, days since last contact, previous contacts, outcome of previous campaigns.
- **Target Variable:** `y` (whether the client subscribed to a term deposit).

## Installation

To set up this project, you'll need to install the following Python libraries:
- NumPy
- pandas
- matplotlib
- seaborn
- scikit-learn

You can install these packages via pip:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Usage

1. **Data Loading:** Load the dataset into a pandas DataFrame.
2. **Data Preprocessing:** Clean the data, handle missing values, encode categorical variables, and normalize numerical data.
3. **Exploratory Data Analysis (EDA):** Perform statistical analysis and visualize the data to understand distributions and relationships.
4. **Model Building:** Train multiple machine learning models to predict the outcome.
5. **Evaluation:** Assess model performance using metrics such as accuracy, precision, recall, and AUC-ROC.

### Running the Code
Run the Jupyter notebook `bank_marketing_analysis.ipynb` to go through the preprocessing, EDA, model training, and evaluation steps.

## Results
The project includes comparing several machine learning models. The best performing model based on accuracy and specificity is the Random Forest Classifier, achieving around 93% accuracy.

## Conclusions
The analysis provides insights into factors influencing the subscription to term deposits and identifies the most effective model for prediction. Further enhancements could include tuning model hyperparameters and testing additional algorithms.

## Contributing
Contributions to the project are welcome. Please ensure to update tests as appropriate.

## License
Distributed under the MIT License. See `LICENSE` for more information.
