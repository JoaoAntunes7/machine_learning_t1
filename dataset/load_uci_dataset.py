from ucimlrepo import fetch_ucirepo
import pandas as pd
from pathlib import Path

# fetch dataset 
def load_uci_dataset(CSV_PATH, ucirepo_id=222):
    if(not Path(CSV_PATH).exists()):
        bank_marketing = fetch_ucirepo(id=ucirepo_id) 
        
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
        df.to_csv(CSV_PATH, index=False)