#linear regression
import streamlit as st
import pandas as p

from sklearn.model_selection import train_split
from sklearn.linear_model import LinearRegression

df=p.read_csv('linearregression.cvs')
x=df['attendance']
y=df['testscrore']
x_train,x_test,y_train,y_test==train_test_split(x,y,test_size=0.2,random_state=15)
model=LinearRegression()
model.fit(x_train,y_train)
st.title('prediction of test score on basis of mobile usage')
st.write('enter your usage of mobile in hours')
hrs=st.number_input('mobile use;')
if st.button('predict score'):
    predict_score=model.predict([[hours]])[0]
    st.success(f"predicted score:{predict_score:2.f}")
st.write("sample trainnig data")
st.dataframe(df)

