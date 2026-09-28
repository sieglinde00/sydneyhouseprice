from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from housing import INPUT_COLUMNS, predict_properties

# Streamlit app for estimating Sydney property sale prices.
ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title='Sydney Property Estimate', layout='wide')
st.markdown('''<style>.block-container{max-width:1120px;padding-top:2.5rem}h1{letter-spacing:-1.5px}div[data-testid="stMetric"]{background:#edf6f7;padding:18px;border-radius:12px}div[data-testid="stMetricValue"]{color:#126779}footer{visibility:hidden}</style>''',unsafe_allow_html=True)
st.caption('MACHINE LEARNING MINI PROJECT  /  SEPTEMBER 2026')
st.title('Sydney Property Estimate')
st.write('Explore an estimated sale price for a property in **Bondi, Chatswood or Parramatta**.')

# Load the trained model and metadata.
@st.cache_resource
def load_model():
    return joblib.load(ROOT/'models/housing.joblib')

bundle = load_model()
meta = bundle['metadata']

# Prediction and context display functions.
def predict(frame):
    return predict_properties(bundle, frame)

# User interface for single property entry and CSV upload.
def show_context(frame):
    flags=[]
    for col,(low,high) in bundle['training_ranges'].items():
        values=pd.to_numeric(frame[col],errors='coerce')
        if ((values<low)|(values>high)).any():
            flags.append(f'{col.replace("_", " ")} is outside the development sample range ({low:g}–{high:g})')
    if frame.property_type.isin(['Block of units','Studio','Villa']).any():
        flags.append('this property type is rare or absent in training')
    if flags:
        st.warning('Extra caution: '+'; '.join(flags)+'.')

# Main app logic for property entry and CSV upload.
mode=st.radio('Choose how to enter properties',['Single property','Upload CSV'],horizontal=True)
if mode == 'Single property':
    with st.form('property'):
        a,b,c=st.columns(3)
        suburb=a.selectbox('Suburb',['Bondi','Chatswood','Parramatta'])
        kind=b.selectbox('Property type',['Apartment','House','Unit','Townhouse','Duplex/semi-detached','Villa','Studio','Block of units'])
        a,b,c,d=st.columns(4)
        beds=a.number_input('Bedrooms',0,20,2)
        baths=b.number_input('Bathrooms',0,20,1)
        parking=c.number_input('Parking spaces',0,20,value=None,placeholder='Unknown')
        area=d.number_input('Advertised area (m²)',min_value=0.0,max_value=10000.0,value=None,placeholder='Unknown')
        text=st.text_area('Agent description (optional)',placeholder='Paste the opening of the listing description. Remove prices.',height=90)
        st.caption('The study uses at most 24 description words per property; the app applies the same limit.')
        submit=st.form_submit_button('Estimate sale price',type='primary',use_container_width=True)
    if submit:
        row=pd.DataFrame([dict(suburb=suburb,property_type=kind,bedrooms=beds,bathrooms=baths,parking=parking,advertised_area_m2=area,description_excerpt=' '.join(text.split()[:24]))])
        try:
            result=predict(row)[0]
            a,c=st.columns(2)
            a.metric('Estimated sale price',f'A${result:,.0f}')
            c.metric('Model',meta['best_model'])
            show_context(row)
        except ValueError as e:
            st.error(str(e))
else:
    example=pd.DataFrame([dict(suburb='Bondi',property_type='Apartment',bedrooms=2,bathrooms=1,parking=1,advertised_area_m2=np.nan,description_excerpt='Light filled apartment with balcony')])
    st.download_button('Download CSV template',example.to_csv(index=False),'property_template.csv','text/csv')
    st.caption('Use the template column names. Blank numeric cells mean unknown; a zero means an actual zero. Up to 1,000 rows.')
    file=st.file_uploader('Upload property features',type=['csv'])
    if file:
        try:
            frame=pd.read_csv(file)
            if 'description_excerpt' in frame:
                frame['description_excerpt']=frame.description_excerpt.fillna('').astype(str).map(lambda s:' '.join(s.split()[:24]))
            values=predict(frame)
            result=frame.copy();result['predicted_sale_price_aud']=np.round(values)
            st.success(f'Estimated {len(result)} properties.')
            st.dataframe(result,hide_index=True,use_container_width=True)
            st.download_button('Download predictions',result.to_csv(index=False),'predictions.csv','text/csv')
            show_context(frame)
        except (ValueError,UnicodeError,pd.errors.ParserError) as e:
            st.error(f'Please check the CSV: {e}')
