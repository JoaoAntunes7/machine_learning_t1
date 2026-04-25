from ucimlrepo import fetch_ucirepo
import pandas as pd
# fetch dataset 
bank_marketing = fetch_ucirepo(id=222) 
  
# data (as pandas dataframes) 
X = bank_marketing.data.features 
y = bank_marketing.data.targets 
  
# metadata 
print(bank_marketing.metadata) 
  
# variable information 
print(bank_marketing.variables) 


# junta features + target em um único DataFrame
df = pd.concat([X, y], axis=1)

# salva em CSV
df.to_csv("bank_marketing.csv", index=False)