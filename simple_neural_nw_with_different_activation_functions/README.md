# Titanic Neural Network Classifier

## Project Structure
- **train.csv**: Training dataset containing features and labels.
- **test.csv**: Test dataset (optional, not used in this project).
- **titanic_nn.ipynb** or **main.py**: Python script or Jupyter Notebook where I implemented the code for preprocessing, model building, and evaluation.
- **README.md**: This file.

## Installation
To run this project, install the following Python libraries:

```bash
pip install tensorflow pandas scikit-learn matplotlib seaborn
```

## How I Built the Model
### Data Preprocessing
- Handled missing values.
- Encoded categorical variables.
- Normalized numerical features to prepare the data for training.

### Model Building
I built three neural networks with different activation functions in the hidden layer:
- **ReLU**
- **Sigmoid**
- **Tanh**

### Training
- Trained each model for **100 epochs**.
- Evaluated their performance on a test set.

### Visualization
- Plotted accuracy and loss over time to track the training progress.

## How to Run the Code
### 1. Download the Dataset
- Download the **train.csv** file from the Kaggle Titanic Dataset.
- Place it in the project directory.

### 2. Run the Script
- **If using a Jupyter Notebook**, open `titanic_nn.ipynb` and execute the cells sequentially.
- **If using a Python script**, run the following command:

```bash
python main.py
```

### 3. Visualize Results
After running the script:
- You'll see the test accuracy for each model.
- Plots will show the training progress.

## Key Features
- **Data Preprocessing**: Handled missing values, encoded categorical variables, and normalized numerical features.
- **Neural Network Models**: Built and trained three neural networks with different activation functions:
  - **ReLU**
  - **Sigmoid**
  - **Tanh**
- **Performance Comparison**: Compared test accuracy and visualized training progress using Matplotlib.

## Results
After running the script, I observed the following test accuracies:

```plaintext
ReLU Model Test Accuracy: 0.82
Sigmoid Model Test Accuracy: 0.79
Tanh Model Test Accuracy: 0.80
```

### Training Progress
(Replace `path_to_plot_image.png` with the actual path to your saved plot image if needed.)

From my experiments, I found that the **ReLU** activation function performed better than **Sigmoid** and **Tanh**, likely due to its ability to mitigate the vanishing gradient problem.

## What I Learned
Through this project, I learned:
- How to preprocess and clean real-world datasets.
- The impact of different activation functions on model performance.
- How to visualize training progress and interpret results.
- The importance of feature engineering and normalization in improving model accuracy.

## Future Improvements
In the future, I plan to:
- Experiment with **deeper neural networks** or additional layers.
- Use advanced techniques like **dropout, batch normalization, or hyperparameter tuning**.
- Apply **cross-validation** for more robust evaluation.
- Explore other machine learning models (e.g., **Random Forest, Gradient Boosting**) for comparison.

## License
This project is open-source and available under the **MIT License**. Feel free to use, modify, and distribute it as needed.
