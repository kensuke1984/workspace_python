import numpy as np
import pandas as pd

d1=pd.Series([1, 2, 3, 5])
print(d1)

df=pd.DataFrame({
    'pi':['Fred'],
    'co':['An'],
    'me':['KK']
})
print(df)
print(df.dtypes)
print(df.columns)

df2=pd.DataFrame(np.random.randn(20, 2))
print(df2)
print(df2.head(6).append(df2.tail(3)))

df3 = pd.DataFrame([[1,2],[3,4]],index=['row 1','row 2'],columns=['col 1','col 2'])

print(df3.loc['row 1',:])
print(df3.iloc[1:2])