import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]}, index=[10, 20, 30])
scaler = StandardScaler()
scaler.set_output(transform="pandas")
df_scaled = scaler.fit_transform(df)
print(df_scaled.index)
