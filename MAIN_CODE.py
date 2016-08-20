# -*- coding: utf-8 -*-
import os
os.chdir(u'E:\工作\实习\嘉实基金产品部\高分红低估值策')

import pandas as pd
import numpy as np
from datetime import timedelta,datetime

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.style.use('ggplot')
%matplotlib inline

import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
from plotly.graph_objs import *
init_notebook_mode() 

ST_CHANGE = pd.read_csv('ST_change.csv',encoding = 'gb2312')

ST_CHANGE

def st_status(x,stock_name,ipo,tuishi):
    ST_status = pd.Series(name = stock_name)
    ST_status[pd.to_datetime(ipo,format = '%Y/%m/%d')] = 'NM'
    if x != u'0':
        for i in x.split(','):
            a,b = i.split(u'：')
            ST_status[pd.to_datetime(b)] = a
    if tuishi != u'0':
        ST_status[pd.to_datetime(tuishi,format = '%Y/%m/%d')] = 'DL'
    return ST_status
    
ST_STATUS = pd.DataFrame()
for i in ST_CHANGE.index:
    x = ST_CHANGE['Hat change'][i]
    stock_name = ST_CHANGE['Code'][i]
    ipo = ST_CHANGE['IPO'][i]
    tuishi = ST_CHANGE['Tuishi'][i]
    ST_status = st_status(x, stock_name,ipo,tuishi)
    ST_STATUS = pd.concat([ST_STATUS, ST_status], axis = 1)

ST_STATUS

SET = set()
for i in ST_STATUS.columns:
    Set = set(ST_STATUS[i].dropna().unique())
    SET = SET.union(Set)

for x in SET:
    print x

ST_STATUS.replace(u'去ST','NM',inplace=True)
ST_STATUS.replace(u'去*ST','NM',inplace=True)
ST_STATUS.replace(u'股票恢复上市','NM',inplace=True)
ST_STATUS.replace(u'*ST变ST','ST',inplace=True)
ST_STATUS.replace(u'股票暂停上市','SP',inplace=True)

zy = ST_STATUS.sort_index().fillna(method='ffill').resample('A').last()['2005-12-31':'2015-12-31']

zy

#Div = pd.read_csv('Yearly_Div.csv',index_col=0)
#Div.index = pd.to_datetime(Div.index)
div_ratio = pd.read_csv('dividend_yield.csv',index_col=0)
div_ratio.index = pd.to_datetime(div_ratio.index)

Price = pd.read_csv('Yearly_Price.csv',index_col=0).replace(0.,np.nan)
Price.index = pd.to_datetime(Price.index)

div_ratio

weighted_divs = pd.DataFrame(index = div_ratio.columns)
for y in range(2005,2016):
    last_day = div_ratio.loc['%d'%y].index[-1]
    weighted_div = (pd.Series([0.2,0.3,0.5],index = div_ratio.loc['%d'%(y-2):'%d'%y].index)).dot(div_ratio.loc['%d'%(y-2):'%d'%y])
    weighted_divs[last_day] = weighted_div

weighted_divs = weighted_divs.T
weighted_divs  = pd.DataFrame(np.where(zy=='NM',weighted_divs,np.nan), index = weighted_divs.index, columns = weighted_divs.columns)   
#只选出不是ST的股票

weighted_divs

div_scores = pd.DataFrame(index = weighted_divs.columns)
for i in weighted_divs.index:
    temp = weighted_divs.ix[i].dropna()
    band = len(temp) / 4.
    q1 = temp.sort_values()[:int(band)].index
    q2 = temp.sort_values()[int(band):int(band)*2].index
    q3 = temp.sort_values()[int(band)*2:int(band)*3].index
    q4 = temp.sort_values()[int(band)*3:].index
    
    score = pd.Series(index = temp.index)
    score[q1] = 1
    score[q2] = 2
    score[q3] = 3
    score[q4] = 4
    div_scores[i] = score
    
div_scores = div_scores.T

div_scores 

Industry = pd.read_csv('Industry_Code.csv',index_col=0).replace(0,np.nan)

Industry = pd.read_csv('industry_wd.csv',index_col = 0).T  #wind行业
Industry.index = pd.to_datetime(Industry.index)

Industry = Industry.replace('0',np.nan)
Industry

industry_index = pd.read_csv('industry_index_wd.csv',encoding='gb2312',header = None)

industry_index

def setScore(available_pe,available_ind):
    Score = pd.Series()
    for ind,stocks in available_pe.groupby(available_ind):
        if ind == np.nan:
            continue
        band = len(stocks) / 4.
        
        q1 = stocks.sort_values()[:int(band)].index
        q2 = stocks.sort_values()[int(band):int(band*2)].index
        q3 = stocks.sort_values()[int(band *2):int(band*3)].index
        q4 = stocks.sort_values()[int(band *3):int(band*4)].index
        
        score = pd.Series(index=stocks.index)
        score[q1] = 4
        score[q2] = 3
        score[q3] = 2
        score[q4] = 1
        Score = pd.concat([Score,score])
    Score.name = available_pe.name
    return Score

PE = pd.read_csv('Yearly_PE.csv',index_col=0)
PE.index = pd.to_datetime(PE.index)

PE

PE_modify = pd.DataFrame(np.where(PE<=0,np.inf,PE),index = PE.index,columns = PE.columns)

PE_score = pd.DataFrame(index=PE.columns)
for i in range(len(PE.index)):
    available = zy.ix[i][zy.ix[i] == 'NM'].index  #选出当时的非ST的上市（去除ST，暂定上市，退市的股票）
    available_pe = PE_modify.ix[i][available]
    available_ind = Industry[available].ix[i] #找到当时上市公司的行业分布
    
    Score = setScore(available_pe,available_ind)
    PE_score[PE.index[i]] = Score

PE_scores = PE_score.T

PE_scores

Ind_PE = pd.read_csv('Industry_PE_wd.csv',index_col=0)
Ind_PE.index = pd.to_datetime(Ind_PE.index)
Ind_PE = pd.DataFrame(np.where(Ind_PE<=0,np.inf,Ind_PE),index = Ind_PE.index,columns = Ind_PE.columns)

Ind_PE

Ind_Scores = pd.DataFrame(index=Ind_PE.columns)
for y in range(2005,2016):
    quantiles = Ind_PE['%d'%(y-2):'%d'%y].quantile([0.25,0.5,0.75])
    last_day = Ind_PE['%d'%y].index[-1]
    pe_last_day = Ind_PE.loc[Ind_PE['%d'%y].index[-1]]
    
    Ind_Score = pd.Series(index=Ind_PE.columns)
    Ind_Score[pe_last_day <= quantiles.loc[0.25]]=4
    Ind_Score[(pe_last_day > quantiles.loc[0.25]) & (pe_last_day <= quantiles.loc[0.5])]=3
    Ind_Score[(pe_last_day > quantiles.loc[0.5]) & (pe_last_day <= quantiles.loc[0.75])]=2
    Ind_Score[pe_last_day > quantiles.loc[0.75]]=1
    
    Ind_Scores[last_day] = Ind_Score

Ind_Scores = Ind_Scores.T
Ind_Scores.index = div_ratio.index[2:]
Ind_Scores
PE_hist = pd.DataFrame(index=Ind_Scores.index,columns=ST_STATUS.columns)
for t in Industry.index:
    for i in SET:
        PE_hist.ix[t][(Industry.ix[t] == i)] = Ind_Scores.ix[t][i]

PE_hist

Scores = div_scores + PE_scores + PE_hist

daily_price = pd.read_csv('Daily_Price.csv',index_col=0).replace(0.,np.nan)
daily_price.index = pd.to_datetime(daily_price.index)

daily_price = daily_price.fillna(method='ffill')
returns = daily_price.pct_change()['2006-01-04':]
mv = pd.read_csv('mv.csv',index_col=0)
daily_mv = pd.read_csv('daily_mv.csv',index_col = 0).replace(0,np.nan)
daily_mv.index = pd.to_datetime(daily_mv.index)
daily_mv = daily_mv['2006-01-04':]
HS300return = pd.read_csv('HS300return.csv',index_col=0)
HS300return.index = pd.to_datetime(HS300return.index)
year_end=list(Scores.index)
year_end.append(pd.to_datetime('2016-12-31'))
name = pd.read_csv('industry_name.csv',encoding ='gb2312',header=None,index_col=1)
industry_name = name[0].drop_duplicates()


def portfolio(daily_price,Scores,n,year_end,FoL=False):
    port_value = pd.Series()
    buy_code = pd.Series()
    buy_weights = pd.Series()
    for i in range(len(Scores.index)):
        start = str(year_end[i]+timedelta(1))[:10]
        end = str(year_end[i+1])[:10]
        available = daily_price[start:end].ix[0].dropna().index
        buy = Scores.ix[i][available].sort_values(ascending=FoL)[:n].index
        
        weights = pd.Series(np.ones_like(buy),index=buy,name=start)/float(n)
        if i > 0:
            shares = port_value[-1] / n / daily_price[buy][start:].ix[0]
        else:
            shares = 1./n / daily_price[buy][start:].ix[0]
        temp = daily_price[buy][start:end].dot(shares)
        
        buy_code = pd.concat([buy_code,pd.Series(buy,name=start)],axis=1)
        buy_weights = pd.concat([buy_weights,weights],axis=1)
        port_value = pd.concat([port_value,temp])
    del buy_code[0]
    del buy_weights[0]
    return [port_value,buy_code,buy_weights]

port_value50,buy50,buy_weights50 = portfolio(daily_price,Scores,50,year_end)
port_value100,buy100,buy_weights100 = portfolio(daily_price,Scores,100,year_end)
port_value150,buy150,buy_weights150 = portfolio(daily_price,Scores,150,year_end)

port_value_50,buy_50,buy_weights_50 = portfolio(daily_price,Scores,50,year_end,True)
port_value_100,buy_100,buy_weights_100 = portfolio(daily_price,Scores,100,year_end,True)
port_value_150,buy_150,buy_weights_150 = portfolio(daily_price,Scores,150,year_end,True)

iplot({'data':[Scatter(x=port_value50.index, y=port_value50/port_value50[0], mode='line',name='First 50'),
               Scatter(x=port_value_50.index, y=port_value_50/port_value_50[0], mode='line',name='Last 50'),
               Scatter(x=returns.index,y=(1+returns.mean(axis=1)).cumprod(),mode='line',name='Simple Average'),
               Scatter(x=HS300return.index,y=(HS300return['000300.SH']+1).cumprod(),mode='line',name='HS300')],
       'layout':Layout(title='Cumulative Returns')})

