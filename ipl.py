try:
    import pandas as pd
    import numpy as np
    match=pd.read_csv('matches.csv')
    delivery=pd.read_csv('deliveries.csv')
    print(match.columns)
    print(delivery.columns)
    #every match has 2 innings.we want total runs of each innings
    total_score_df=delivery.groupby(['match_id','inning']).sum()['total_runs'].reset_index()
    #print(total_score_df)
    total_score_df=total_score_df[total_score_df['inning']==1]
    print(total_score_df)
    #marge this data into matches dataset
    match_df=match.merge(total_score_df[['match_id','total_runs']],left_on="id",right_on="match_id")
    print(match_df.head())
    print('unique',match_df['team1'].unique())
    teams=['Sunrisers Hyderabad', 'Mumbai Indians', 'Gujarat Lions' ,'Royal Challengers Bangalore',
    'Kolkata Knight Riders',  'Kings XI Punjab',
    'Chennai Super Kings' ,'Rajasthan Royals' ,
  
    'Delhi Capitals']
    match_df['team1']=match_df['team1'].str.replace('Delhi Daredevils','Delhi Capitals')
    match_df['team2']=match_df['team2'].str.replace('Delhi Daredevils','Delhi Capitals')
    match_df['team1']=match_df['team1'].str.replace('Deccan Chargers','Sunrisers Hyderabad')
    match_df['team2']=match_df['team2'].str.replace('Deccan Chargers','Sunrisers Hyderabad')
    match_df=match_df[match_df['team1'].isin(teams)]
    match_df=match_df[match_df['team2'].isin(teams)]
    print(match_df.shape)
    #dl value count
    print('value count',match_df["dl_applied"].value_counts())
    match_df=match_df[match_df["dl_applied"]==0]
    print(match_df.head())
    match_df=match_df[['match_id','city','winner','total_runs']]
    #join this table with another table
    delivery_df=match_df.merge(delivery,on='match_id')
    print(delivery_df.head())
#   now i need innings value 2
    delivery_df=delivery_df[delivery_df['inning']==2]
    print(delivery_df.head())
    delivery_df['current_score']=delivery_df.groupby('match_id').cumsum()['total_runs_y']
    print(delivery_df.head())
    delivery_df['runs_left']=(delivery_df['total_runs_x']+1)-delivery_df['current_score']
    print(delivery_df.head())
    delivery_df['balls_left']=126-(delivery_df['over']*6+delivery_df['ball'])
    print(delivery_df.head())
#wicket
#problem
    delivery_df['player_dismissed']=delivery_df['player_dismissed'].fillna("0")
    delivery_df['player_dismissed']=delivery_df['player_dismissed'].apply(lambda x:x if x=="0" else "1")
    delivery_df['player_dismissed']=delivery_df['player_dismissed'].astype('int')
    wickets=delivery_df.groupby('match_id').cumsum()['player_dismissed'].values
    delivery_df['wickets']=10-wickets
    #print(delivery_df.head())
    print(delivery_df.tail(15))
    #current run rate/crr=runs/overs
    delivery_df['crr']=(delivery_df['current_score']*6)/(120-delivery_df['balls_left'])
    #required run rate/rrr=left runs/left over
    delivery_df['rrr']=(delivery_df['runs_left']*6)/delivery_df['balls_left']
    print(delivery_df.head())
    #result(if batting team and winner is same then 1 else 0)
    def result(row):
        return 1 if row['batting_team']== row['winner'] else 0 
    delivery_df['result']=delivery_df.apply(result,axis=1) 
#extract needed columns
    final_df=delivery_df[['batting_team','bowling_team','city','runs_left','balls_left','wickets','total_runs_x','crr','rrr','result']]
    print(final_df.head())
#we need suffeling as for biased
    final_df=final_df.sample(final_df.shape[0])
    print(final_df.sample())
    print(final_df.isnull().sum())
    final_df.dropna(inplace=True)
    final_df=final_df[final_df['balls_left']!=0]
#=============================================================================================================================================
    x=final_df.drop('result',axis=1)
    y=final_df[['result']]
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)
    print(x_train.shape)
    trf=pd.get_dummies(final_df,columns=['batting_team','bowling_team','city'])
    print(trf.head())
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    trf=ColumnTransformer([('trf',OneHotEncoder(sparse=False,drop='first'),['batting_team','bowling_team','city'])],remainder='passthrough')

#===========================
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    pipe=Pipeline(steps=[('step1',trf),
                     ('step2',LogisticRegression(solver='liblinear'))])
    pipe.fit(x_train,y_train)
    
    #print(x_train.describe())
    y_pred=pipe.predict(x_test)
    #=============
    from sklearn.metrics import accuracy_score
    print(accuracy_score(y_test,y_pred))

    piper=Pipeline(steps=[('step1',trf),('step2',RandomForestClassifier())])
    piper.fit(x_train,y_train)
    y_pred2=piper.predict(x_test)
    print(accuracy_score(y_test,y_pred2))
    print(pipe.predict_proba(x_test)[1])
    print(piper.predict_proba(x_test)[1])
#=================
    import pickle
    pickle.dump(pipe,open('pipe.pkl','wb'))
    print(teams)
    print(delivery_df['city'].unique())
   
    
except Exception as e:
    print(e)
    
 

