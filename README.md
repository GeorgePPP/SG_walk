# Featues explaination

| Feature             | Type (Category/Continuous) | Explanation                                           |
| ------------------- | ------------------------- | ----------------------------------------------------- |
| Participant ID      | Integer, Categorical      | Identifier for each participant                       |
| Game Type            | String, Categorical       | Types of games played by participants (4 categories)  |
| Week                | Integer, Categorical      | Week in a month (categorical representation)           |
| Calories Burnt       | Float, Continuous         | Amount of calories burnt                               |
| Action Count         | Integer, Continuous       | Count of actions performed                             |
| Max Acceleration     | Float, Continuous         | Maximum acceleration achieved during a game            |
| Grouping             | Integer, Categorical      | Grouping category (1 = Peer, 2 = HC, 3 = Exercise, 4 = Alone) |
| Gender               | Integer, Categorical      | Gender (0 = Female, 1 = Male)                          |
| Age                 | Integer, Continuous       | Age of the participant                                 |
| Income              | Integer, Categorical      | Income level (1 - 10 categories)                       |
| Education            | Integer, Categorical      | Education level (1 - 10 categories)                    |
| SelfSES              | Integer, Categorical      | Social Economic Status (1 - 10 categories)             |
| Companion            | Integer, Categorical      | Same as Grouping category                              |
| Emotional           | Integer, Categorical      | Emotional well-being score (1 - 10 categories)         |
| Psychological        | Integer, Categorical      | Psychological well-being score (1 - 10 categories)    |
| Social              | Integer, Categorical      | Social well-being score (1 - 10 categories)            |
| Languishing          | Integer, Categorical      | State of languishing (0 = No, 1 = Yes)                 |
| Flourishing          | Integer, Categorical      | State of flourishing (0 = No, 1 = Yes)                |
| Moderate            | Integer, Categorical      | State of moderate well-being (0 = No, 1 = Yes)        |

## Machine Learning Model Comparison

| Aspect                | Logistic Regression | Random Forest      | Naive Bayes        | Support Vector Machine (SVM) |
|-----------------------|---------------------|--------------------|--------------------|-------------------------------|
| Accuracy              | Suitable for linearly separable datasets. May not perform well if features have complex interactions. | Generally performs well, handles non-linearity effectively, and can capture complex relationships. | Simple and may perform well with small datasets if the independence assumption holds. May not capture complex relationships well. | Can handle both linear and non-linear datasets effectively. Performance depends on kernel choice and hyperparameters. |
| Latency               | Low computational cost since it's a linear model. Fast training and prediction. | Moderate computational cost due to the ensemble of decision trees. Slower training than logistic regression. | Low computational cost since it's based on simple probability calculations. Fast training and prediction. | Moderate to high computational cost, especially with non-linear kernels. Slower training and prediction compared to the others. |
| Interpretability      | Highly interpretable. Coefficients indicate the impact of each feature on the target. | Less interpretable due to ensemble nature, but feature importance can be calculated. | Highly interpretable since it relies on probability and simple calculations. Easy to understand the conditional probabilities. | Less interpretable than logistic regression and naive Bayes. The decision boundary can be complex in high dimensions. |
| Robustness (40 samples)| Prone to overfitting with a small dataset, especially if features are highly correlated or have complex interactions. Regularization can help. | Relatively robust due to the ensemble of trees. Less prone to overfitting than decision trees. | Can perform well with small datasets, but sensitive to the independence assumption. May not capture complex relationships well. | Can be sensitive to the choice of kernel and hyperparameters. Cross-validation and parameter tuning are crucial. |

