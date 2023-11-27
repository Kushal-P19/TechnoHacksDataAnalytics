import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use("/tmp/rose-pine-dawn.mplstyle")

pd.options.display.max_columns=100
pd.options.display.max_rows=100
path = "/kaggle/input/house-prices-advanced-regression-techniques/"
train = pd.read_csv(path+"train.csv").drop("Id",axis=1)
test = pd.read_csv(path+"test.csv").drop("Id",axis=1)
sub = pd.read_csv(path+"sample_submission.csv")


def plot_target(train, col, title, pie_colors):
    pass


plot_target(train,
            col="SalePrice_Range",
            title="SalePrice",
            pie_colors=["#abc9ea","#98daa7","#f3aba8","#d3c3f7","#f3f3af","#c0ebe9"])

target = "SalePrice"
plt.figure(figsize=(14,len(cat_cols)*3))
for idx,column in enumerate(cat_cols ):
    data = df[df["group"] == "train"].groupby(column)[target].mean().reset_index().sort_values(by=target)
    plt.subplot(len(cat_cols)//2+1,2,idx+1)
    sns.barplot(y=column, x=target, data=data, palette="pastel")


    def con_cat(train, test):
        df1, df2 = train.copy(), test.copy()
        df1["group"] = "train"
        df2["group"] = "test"

        return pd.concat([df1, df2], axis=0, ignore_index=True)


    def find_col_dtypes(data, ord_th):
        num_cols = data.select_dtypes("number").columns.to_list()
        cat_cols = data.select_dtypes("object").columns.to_list()

        ordinals = [col for col in num_cols if data[col].nunique() < ord_th]

        num_cols = [col for col in num_cols if col not in ordinals]

        return num_cols, ordinals, cat_cols


    num_cols, ordinals, cat_cols = find_col_dtypes(test, 20)

    print(f"Num Cols: {num_cols}", end="\n\n")
    print(f"Cat Cols: {cat_cols}", end="\n\n")
    print(f"Ordinal Cols: {ordinals}")


    df = con_cat(train, test)
    for p, count in enumerate(data[target].values,0):
        plt.text(count + 10, p+0.05/len(data), f"{int(count//1000)}k", color='black', fontsize=11)
    plt.title(f"{column} and {target}")
    plt.xticks(fontweight='bold')
    plt.box(False)
    plt.tight_layout()

plt.figure(figsize=(14,len(num_cols)*3))
for idx,column in enumerate(num_cols):
    plt.subplot(len(num_cols)//2+1,2,idx+1)
    sns.boxplot(x="SalePrice_Range", y=column, data=train,palette="pastel")
    plt.title(f"{column} Distribution")
    plt.tight_layout()

plt.figure(figsize=(12,10))
corr=df[num_cols].corr(numeric_only=True)
mask= np.triu(np.ones_like(corr))
sns.heatmap(corr, annot=True, fmt=".1f", linewidths=1, mask=mask, cmap=sns.color_palette("vlag"));

