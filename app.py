
import streamlit as st
import pickle
import pandas as pd
teams=['Sunrisers Hyderabad', 'Mumbai Indians', 'Gujarat Lions', 'Royal Challengers Bangalore', 'Kolkata Knight Riders', 'Kings XI Punjab', 'Chennai Super Kings', 'Rajasthan Royals', 'Delhi Capitals']
citys=['Hyderabad', 'Rajkot' ,'Bangalore' ,'Mumbai', 'Indore', 'Kolkata', 'Delhi', 'Chandigarh', 'Kanpur', 'Jaipur', 'Chennai', 'Cape Town' ,'Port Elizabeth' ,'Durban' ,'Centurion', 'East London', 'Johannesburg' ,'Kimberley',
'Bloemfontein', 'Ahmedabad' ,'Cuttack', 'Nagpur', 'Dharamsala',
'Visakhapatnam' ,'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi', 'Sharjah', 
'Mohali', 'Bengaluru']
pipe=pickle.load(open('D:\ipl\pipe.pkl','rb'))
st.title('IPL Win Predictor')
col1,col2=st.columns(2)
#try:
with col1:
    batting_team=st.selectbox('Select the batting team',sorted(teams))
with col2:
    bowling_team=st.selectbox('Select the bowling team',sorted(teams))
selected_city=st.selectbox('select host city',sorted(citys))
target=st.number_input('Target')
col3,col4,col5=st.columns(3)
with col3:
    Score=st.number_input('Score')
with col4:
    overs=st.number_input('Overs completed')
with col5:
    wickets=st.number_input('wickets out')
if st.button('predict probability'):
    runs_left=target-Score
    balls_left=120-(overs*6)
    wickets=10-wickets
    crr=Score/overs
    rrr=(runs_left*6)/balls_left
    input_df=pd.DataFrame({'batting_team':[batting_team],'bowling_team':[bowling_team],'city':[selected_city],'runs_left':[runs_left],'balls_left':[balls_left],'wickets':[wickets],'total_runs_x':[target],'crr':[crr],'rrr':[rrr]})
    result=pipe.predict_proba(input_df)
    loss=result[0][0]
    win=result[0][1]
    st.text(batting_team + "_"+str(round(win*100))+"%")
    st.text(bowling_team + "_"+str(round(loss*100))+"%")
#except Exception as e:
  #  print(e)
    

