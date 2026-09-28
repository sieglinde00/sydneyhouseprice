from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

# Compare valuations from machine learning model, LLM, and human estimates against actual sale prices.
OUT=Path('outputs')
KEYS=['address','suburb']
human=pd.read_csv(OUT/'human_estimates_blind.csv')
llm=pd.read_csv(OUT/'llm_estimates.csv')
predictions=pd.read_csv(OUT/'test_predictions.csv')

# Ensure that human and LLM estimates refer to the same properties and that predictions exist for comparison.
if set(map(tuple,human[KEYS].to_numpy()))!=set(map(tuple,llm[KEYS].to_numpy())):
    raise ValueError('Human and LLM estimates must refer to the same ten addresses and suburbs.')
key=human[KEYS].merge(predictions[KEYS+['sale_price','ml_estimate']],on=KEYS,how='left',validate='one_to_one')
if key[['sale_price','ml_estimate']].isna().any().any():
    raise ValueError('Every comparison property must match a held-out prediction by address and suburb.')
entered=human.human_estimate.astype(str).str.strip().replace({'nan':'','None':''})
human['human_estimate']=pd.to_numeric(entered.str.replace(r'[A$ ,]','',regex=True),errors='coerce')
table=key.merge(llm,on=KEYS,validate='one_to_one').merge(human[KEYS+['human_estimate']],on=KEYS,validate='one_to_one')

# Compute absolute errors and summary metrics for each valuation approach.
records=[]
for col in ['ml_estimate','llm_estimate','human_estimate']:
    table[col+'_absolute_error']=abs(table[col]-table.sale_price)
    valid=table[col].notna()
    record={'approach':col,'n':int(valid.sum()),'status':'complete' if valid.all() else 'pending human input'}
    if valid.all():
        record.update(MAE=mean_absolute_error(table.sale_price,table[col]),RMSE=np.sqrt(mean_squared_error(table.sale_price,table[col])),MAPE_pct=100*mean_absolute_percentage_error(table.sale_price,table[col]))
    records.append(record)
table.to_csv(OUT/'valuation_comparison.csv',index=False)
pd.DataFrame(records).to_csv(OUT/'valuation_metrics.csv',index=False)
print(pd.DataFrame(records).to_string(index=False))
