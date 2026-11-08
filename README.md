# 🌸 Iris Flower Classification – Machine Learning Project

A beginner machine learning project in Python that predicts the species of an iris flower (**Setosa, Versicolor or Virginica**) from four measurements: sepal length, sepal width, petal length and petal width.

## 📂 Files
| File | Description |
|------|-------------|
| `iris_classification.ipynb` | Jupyter notebook (run in VS Code or Google Colab) |
| `iris_classification.py` | Same code as a Python script |
| `iris.data` | Iris dataset – 150 rows, 5 columns |

## 🛠️ Tools and Libraries
Python · NumPy · Pandas · Matplotlib · Seaborn · Scikit-learn · VS Code · Google Colab

## 🔄 Project Steps
1. Import libraries
2. Load the dataset with Pandas
3. Explore and visualize the data (`describe()`, Seaborn pairplot)
4. Separate the input features (X) and output label (Y)
5. Split the data into 80% training and 20% testing
6. Train and compare 3 models:
   - Support Vector Machine (SVC)
   - Logistic Regression
   - Decision Tree Classifier
7. Evaluate with accuracy score and classification report
8. Predict the species of new flowers

## 📊 Results
All three models reached about **93–97% accuracy** on the test data. Results vary slightly between runs because the train/test split is random.

## ▶️ How to Run
**Google Colab:** Upload `iris_classification.ipynb`, upload `iris.data` through the folder icon, then click **Runtime → Run all**.

**VS Code:** Put the notebook and `iris.data` in the same folder, install the libraries, then click **Run All**:
```
pip install numpy pandas matplotlib seaborn scikit-learn
```

## 🙏 Credit
Built by following the YouTube tutorial *"Machine learning beginner full project with code and dataset iris flower classification"* by **Let's Crack It!**
