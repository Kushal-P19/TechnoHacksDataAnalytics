import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

training = pd.read_csv("/kaggle/input/titanic/train.csv")
test = pd.read_csv("/kaggle/input/titanic/test.csv")

training['train_test'] = 1
test['train_test'] = 0
test['Survived'] = np.NaN
all_data = pd.concat([training, test])

training.info()
training.head()
training.describe()
training.describe().columns
df_num = training[['Age', 'SibSp', 'Parch', 'Fare']]
df_cat = training[['Survived', 'Pclass', 'Sex', 'Ticket', 'Cabin', 'Embarked']]
for i in df_num.columns:
    plt.hist(df_num[i])
    plt.title(i)
    plt.show()
print(df_num.corr())
sns.heatmap(df_num.corr())
pd.pivot_table(training, index='Survived', values=['Age', 'SibSp', 'Parch', 'Fare'])
for i in df_cat.columns:
    sns.barplot(x=df_cat[i].value_counts().index, y=df_cat[i].value_counts())
    plt.title(i)
    plt.show()
training.groupby(['Ticket'], as_index=False)['Survived'].mean()
print(pd.pivot_table(training, index='Survived', columns='Pclass', values='Ticket', aggfunc='count'))
print()
print(pd.pivot_table(training, index='Survived', columns='Sex', values='Ticket', aggfunc='count'))
print()
print(pd.pivot_table(training, index='Survived', columns='Embarked', values='Ticket', aggfunc='count'))
print()
df_cat.Cabin
training['cabin_multiple'] = training.Cabin.apply(lambda x: 0 if pd.isna(x) else len(x.split(' ')))

training['cabin_multiple'].value_counts()
pd.pivot_table(training, index='Survived', columns='cabin_multiple', values='Ticket', aggfunc='count')
training['cabin_adv'] = training.Cabin.apply(lambda x: str(x)[0])

print(training.cabin_adv.value_counts())  # Corrected method name
pd.pivot_table(training, index='Survived', columns='cabin_adv', values='Name', aggfunc='count')


def is_numeric_ticket(ticket):
    return 1 if ticket.isnumeric() else 0


def process_ticket_letters(ticket):
    ticket_parts = ticket.split(' ')[:-1]

    if len(ticket_parts) > 0:
        cleaned_ticket = ''.join(ticket_parts).replace('.', '').replace('/', '').lower()
    else:
        cleaned_ticket = 0

    return cleaned_ticket


training['numeric_ticket'] = training['Ticket'].apply(is_numeric_ticket)
training['ticket_letters'] = training['Ticket'].apply(process_ticket_letters)
numeric_ticket_counts = training['numeric_ticket'].value_counts()
print(numeric_ticket_counts)

all_data['cabin_multiple'] = all_data.Cabin.apply(lambda x: 0 if pd.isna(x) else len(x.split(' ')))

all_data['cabin_adv'] = all_data.Cabin.apply(lambda x: str(x)[0])
all_data['numeric_ticket'] = all_data.Ticket.apply(lambda x: 1 if x.isnumeric() else 0)
all_data['ticket_letters'] = all_data.Ticket.apply(lambda x: ''.join(x.split(' ')[:-1]).replace('.','')
                                                   .replace('/', '').lower()
                                                    if len(x.split(' ')[:-1]) > 0 else 0)
all_data['name_title'] = all_data.Name.apply(lambda x: x.split(',')[1].split('.')[0].strip())

all_data.Age = all_data.Age.fillna(training.Age.mean())
all_data.Fare = all_data.Fare.fillna(training.Fare.mean())

all_data.dropna(subset=['Embarked'],inplace = True)

all_data['norm_fare'] = np.log(all_data.Fare+1)
all_data['norm_fare'].hist()

all_data.Pclass = all_data.Pclass.astype(str)

all_dummies = pd.get_dummies(all_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'norm_fare', 'Embarked', 'cabin_adv',
                                       'cabin_multiple', 'numeric_ticket', 'name_title', 'train_test']])

X_train = all_dummies[all_dummies.train_test == 1].drop(['train_test'], axis = 1)
X_test = all_dummies[all_dummies.train_test == 0].drop(['train_test'], axis = 1)

y_train = all_data[all_data.train_test == 1].Survived
y_train.shape

