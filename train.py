from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.inspection import permutation_importance
from housing import make_model, INPUT_COLUMNS

# Set up output directories and load the dataset.
OUT = Path('outputs')
OUT.mkdir(exist_ok=True)
(OUT/'figures').mkdir(exist_ok=True)
Path('models').mkdir(exist_ok=True)
dataset_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('data/properties.csv')
df = pd.read_csv(dataset_path, dtype={'address':str}).fillna({'description_excerpt':''})
dev_idx, test_idx = train_test_split(np.arange(len(df)), test_size=0.2, stratify=df.suburb, random_state=42)
dev, test = df.iloc[dev_idx].copy(), df.iloc[test_idx].copy()
X, y = dev[INPUT_COLUMNS], dev.sale_price
Xt, yt = test[INPUT_COLUMNS], test.sale_price
folds = list(KFold(n_splits=5, shuffle=True, random_state=42).split(dev))

# Define evaluation metrics for model performance.
def metrics(actual, predicted):
    return {'MAE':float(mean_absolute_error(actual,predicted)), 'RMSE':float(np.sqrt(mean_squared_error(actual,predicted))),
            'R2':float(r2_score(actual,predicted)), 'MAPE_pct':float(100*mean_absolute_percentage_error(actual,predicted))}

# Cross-validation and out-of-fold predictions for model comparison.
scores, oof = [], {}
names = ['Median baseline','Ridge','Decision tree','Random forest']
for name in names:
    pred = np.zeros(len(dev))
    for k,(tr,va) in enumerate(folds,1):
        model = make_model(name).fit(X.iloc[tr],y.iloc[tr])
        pred[va] = model.predict(X.iloc[va])
        scores.append({'model':name,'fold':k,**metrics(y.iloc[va],pred[va]),
                       'train_MAE':mean_absolute_error(y.iloc[tr],model.predict(X.iloc[tr]))})
    oof[name] = pred
cv = pd.DataFrame(scores)
summary = cv.groupby('model').agg(MAE=('MAE','mean'),MAE_sd=('MAE','std'),RMSE=('RMSE','mean'),R2=('R2','mean'),MAPE_pct=('MAPE_pct','mean'),train_MAE=('train_MAE','mean')).sort_values('MAE')
summary.to_csv(OUT/'cv_summary.csv')
best = summary.drop(index='Median baseline').index[0]
model = make_model(best).fit(X,y)
pred = model.predict(Xt)
test['ml_estimate'] = pred
test['signed_error'] = pred - yt
test['absolute_error'] = abs(pred - yt)
test['APE_pct'] = abs(pred - yt)/yt*100
test.to_csv(OUT/'test_predictions.csv',index=False)

# Preserve the original ten comparison cases and recorded human/LLM inputs.
mapping = pd.read_csv(OUT/'comparison_records.csv', dtype={'property_id':str})
ten = mapping.merge(test, on=['address','suburb'], validate='one_to_one')
assert len(ten) == 10, 'Original comparison cases must remain held out.'

# Text ablation is diagnostic only, not used to choose the winner
abl=[]
for k,(tr,va) in enumerate(folds,1):
    no_text = make_model(best,include_text=False).fit(X.iloc[tr],y.iloc[tr])
    abl.append({'fold':k,**metrics(y.iloc[va],no_text.predict(X.iloc[va]))})

# Permutation importance is evaluated within development CV, never on fitted training rows.
imps=[]
for tr,va in folds:
    m=make_model(best).fit(X.iloc[tr],y.iloc[tr])
    pi=permutation_importance(m,X.iloc[va],y.iloc[va],scoring='neg_mean_absolute_error',n_repeats=8,random_state=42)
    imps.append(pi.importances_mean)
importance=pd.DataFrame({'feature':INPUT_COLUMNS,'MAE_increase':np.mean(imps,axis=0),'fold_sd':np.std(imps,axis=0)})
importance.sort_values('MAE_increase',ascending=False).to_csv(OUT/'feature_importance.csv',index=False)

# Compile and save overall metrics, including test performance, out-of-fold MAE, and text ablation results.
info={'best_model':best,'n_total':len(df),'n_development':len(dev),'n_test':len(test),
      'test_metrics':metrics(yt,pred),'dummy_test_metrics':metrics(yt,np.repeat(y.median(),len(yt))),
      'OOF_MAE':mean_absolute_error(y,oof[best]),
      'text_ablation_MAE':float(pd.DataFrame(abl).MAE.mean())}
metric_row={k:v for k,v in info.items() if not isinstance(v,dict)}
for group in ['test_metrics','dummy_test_metrics']:
    metric_row.update({group+'_'+k:v for k,v in info[group].items()})
pd.DataFrame([metric_row]).to_csv(OUT/'metrics.csv',index=False)

# Save the trained model, metadata, and training ranges for streamlit
joblib.dump({'model':model,'metadata':info,'training_ranges':{c:[float(dev[c].min()),float(dev[c].max())] for c in ['bedrooms','bathrooms','parking','advertised_area_m2']}},'models/housing.joblib')

# Exploratory data analysis and visualization.
plt.rcParams.update({'figure.dpi':150,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','font.size':10})
colours={'Bondi':'teal','Chatswood':'purple','Parramatta':'orange'}
fig,axs=plt.subplots(1,3,figsize=(14,4))
axs[0].hist(df.sale_price/1e6,bins=18,color='teal',edgecolor='white');axs[0].set(xlabel='Sale price (AUD millions)',ylabel='Properties',title='Right-skewed sale prices')
axs[1].boxplot([df[df.suburb.eq(s)].sale_price/1e6 for s in colours],tick_labels=list(colours));axs[1].set(ylabel='Sale price (AUD millions)',title='Different market segments')
for s,c in colours.items():
    p=df[df.suburb.eq(s)];axs[2].scatter(p.bedrooms,p.sale_price/1e6,s=20,alpha=.7,label=s,color=c)
axs[2].set(xlabel='Bedrooms',ylabel='Sale price (AUD millions)',title='Accommodation and price');axs[2].legend(fontsize=8)
fig.tight_layout();fig.savefig(OUT/'figures/eda.png');plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(11,4))
pos=np.arange(len(summary));axs[0].bar(pos-.18,summary.train_MAE/1000,.36,label='Training',color='lightblue');axs[0].bar(pos+.18,summary.MAE/1000,.36,yerr=summary.MAE_sd/1000,label='CV validation (SD)',color='teal');axs[0].set_xticks(pos,summary.index,rotation=15);axs[0].set(ylabel='MAE (AUD thousands)',title='Complexity and generalisation');axs[0].legend(fontsize=8)
for s,c in colours.items():
    p=test[test.suburb.eq(s)];axs[1].scatter(p.sale_price/1e6,p.ml_estimate/1e6,color=c,label=s)
limit=max(test.sale_price.max(),test.ml_estimate.max())/1e6*1.05
axs[1].plot([0,limit],[0,limit],'--',color='grey');axs[1].set(xlabel='Actual (AUD millions)',ylabel='Predicted (AUD millions)',title=f'Held-out predictions: {best}');axs[1].legend(fontsize=8)
fig.tight_layout();fig.savefig(OUT/'figures/evaluation.png');plt.close(fig)
fig,ax=plt.subplots(figsize=(8,4));p=importance.sort_values('MAE_increase');ax.barh(p.feature,p.MAE_increase/1000,color='teal');ax.set(xlabel='Increase in validation MAE after permutation (AUD thousands)',title='Predictive importance, not causal effects');fig.tight_layout();fig.savefig(OUT/'figures/importance.png');plt.close(fig)
print(summary.round(2).to_string());print(pd.Series(metric_row).to_string())
