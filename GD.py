import pandas as pd
data=pd.read_csv('assignment1_data.csv')
print(data)
x=data.A
y=data.B
alpha=1e-4

w=w+2*alpha*(r-R*w)