import matplotlib.pyplot as plt
import seaborn as sns

def plot_feature_correlation(df, features):
    corr = df[features].corr()
    plt.figure(figsize=(8,6))
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Feature Correlation Matrix")
    plt.show()
