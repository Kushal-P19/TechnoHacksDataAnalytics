import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")
df = pd.read_csv('/kaggle/input/iris/Iris.csv',index_col='Id')
for i in range(4):
    fig = px.scatter(x=df.iloc[:,i],color=df['Species'])
    fig.update_layout(title=df.columns[i]+ ' for each Species')
    fig.update_xaxes(title_text=df.columns[i])
    fig.update_yaxes(title_text='Measure in CM')
    fig.show()
fig = px.scatter(data_frame=df,x='SepalLengthCm',y='SepalWidthCm' , color='Species')
fig.update_layout(title='Sepal Length vs Sepal Width')
fig.show()
fig = px.scatter(data_frame=df,x='PetalLengthCm',y='PetalWidthCm' , color='Species')
fig.update_layout(title='Petal Length vs Petal Width')
fig.show()
for i in range(4):
    fig = px.histogram(df, x=df.columns[i], color="Species", facet_col="Species", histnorm="probability density", nbins=100)
    fig.update_layout(title="Density distribution of "+df.columns[i]+" by Species")
    fig.show()
fig = px.pie(values=df['Species'].value_counts().values,names=df['Species'].value_counts().index)
fig.update_layout(title="Species")
fig.show()
plt.figure(figsize=(10,5))
sns.heatmap(df.select_dtypes('number').corr(),annot=True)
plt.title('Correlation')
plt.show()
