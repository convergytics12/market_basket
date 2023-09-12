import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import numpy as np
import streamlit as st

st.title('Market Basket Analysis')
st.image('market.jpg',width=800)
f=st.file_uploader('Upload file',key='f3')

if f is not None:
    df=pd.read_excel(f)
    df['Item']=df['Brand']+'_'+df['Sub_Category']
    df=df[['Invoice_ID','Item']]
    for i in range(len(df)):
        df['Item'][i]=df['Item'][i].split('(')[0].rstrip()
    df['Quantity']=1
    df['Invoice_ID'] = df['Invoice_ID'].astype('str')
    
    mybasket=(df.groupby(['Invoice_ID','Item'])['Quantity'].sum().unstack().reset_index().fillna(0).set_index('Invoice_ID'))
    
    
    
    def my_encode_units(x):
        if x <= 0:
            return 0
        if x >= 1:
            return 1

    my_basket_sets = mybasket.applymap(my_encode_units)
    
    my_frequent_itemsets = apriori(my_basket_sets, min_support=0.01, use_colnames=True)
    
    my_rules = association_rules(my_frequent_itemsets, metric="lift", min_threshold=1)
    
    s=my_rules.sort_values("confidence",ascending=False).reset_index(drop=True)
    
    unique_items=df['Item'].unique()
    options=st.multiselect('**Select the items**',['Select an option']+list(unique_items))
    #st.write(options)
    
    res=s.copy()
    res["antecedents"] = res["antecedents"].apply(lambda x: ', '.join(list(x))).astype("unicode")
    res["consequents"] = res["consequents"].apply(lambda x: ', '.join(list(x))).astype("unicode")
    
    fdf=res[res['antecedents'].isin(options)][['antecedents','consequents','confidence','support']]
    
    

    
    
    if(st.button('Submit')):
        progress=st.progress(0)
        for i in range(100):
            #sleep(0.05)
            progress.progress(i+1)
            
        st.subheader('Possible Combinations')
        st.table(fdf)