import numpy as np
from sklearn.preprocessing import OneHotEncoder

from SRacos import SRacos
from framework.base import ModelEvaluator
from framework.gbdt import LightGBM

enc = OneHotEncoder()


def loadDatadet1(infile):
    f = open(infile, 'r')
    sourceInLine = f.readlines()
    dataset = []
    for line in sourceInLine:
        #        line=line.replace("","nan")
        temp1 = line.strip('\n')
        temp2 = temp1.split(',')
        temp2 = list(map(float, temp2))
        dataset.append(temp2)
    return dataset


def loadDatadet2(infile):
    train_y = []
    with open('./temp_dataset/Titanic/train_y.csv', 'r', encoding='utf-8') as f1:
        for line in f1.readlines():
            line = line.strip("\n")
            train_y.append(line)
            train_y = list(map(int, train_y))
    return train_y


infile1 = './temp_dataset/Titanic/train_x2.csv'
train_x = loadDatadet1(infile1)
print(train_x)
infile2 = './temp_dataset/Titanic/train_y.csv'
train_y = loadDatadet2(infile2)
print(train_y)
# preprocessing
"""df = pd.DataFrame(train_x)
#df.columns = ['passenger', 'pclass', 'sex', 'age','sib','parch','ticket','fare','carbin','embarked']
df.columns = ['passenger', 'pclass', 'sex', 'age','sib','parch','fare','embarked']

sex_mapping = {'male': 1,'female': 0}
df['sex'] = df['sex'].map(sex_mapping)

em_mapping = {'S': 1,'C': 0,'Q':2}
df['embarked'] = df['embarked'].map(em_mapping)

df['passenger']=list(df['passenger'].map(int))
df['pclass']=df['pclass'].map(int)
df['age']=df['age'].map(float)
df['sib']=df['sib'].map(int)
df['parch']=df['parch'].map(int)
df['fare']=df['fare'].map(float)

train_x=df"""
print(train_x)
train_x = np.array(train_x)
train_y = np.array(train_y)
print(train_x)
print(train_y)

model = LightGBM()
evaluator = ModelEvaluator(model_generator=model, train_x=train_x, train_y=train_y)

dimension = [param.retrieve_raw_param() for param in model.hp_space]

sracos = SRacos.Optimizer()
x, y = sracos.opt(objective=evaluator.evaluate,
                  dimension=dimension, budget=100, k=3, r=5, prob=0.99, max_coordinates=2, print_opt=True)
print(x, y)
