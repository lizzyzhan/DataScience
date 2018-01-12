# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 08:49:01 2018

@author: 10206913
"""

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import pandas as pd
import numpy as np

#df = pd.read_excel('.\\data\\salesfunnel.xlsx')
df = pd.read_excel('.\\data\\canalys_zte_2017q3_sp_101117_Li.xlsx','97 Database')
year_options = df['Year'].unique()
latest_year=df['Year'].max()

region_options = ['(Total)']+list(df['Region'].unique())
sub_region_options = ['(Total)']+list(df['Sub-region'].unique())
country_options = ['(Total)']+list(df["Country"].unique())
vendor_options = ['(Total)']+list(df['Vendor'].unique())
quarter_options = df['Quarter'].unique()

app = dash.Dash()

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}


app.layout = html.Div([
    html.H1(
        children=' CA DashBoard',
        style={
            'textAlign': 'center',
            'color': colors['text']
        }
    ),
    html.Div([
        html.Div(
            [
                html.H3(
                    children='Select',
                    style={
                        'textAlign': 'center'
                    }
                ),
                html.Label('Year: '),
                dcc.Dropdown(
                    id="Year",
                    options=[{
                        'label': i,
                        'value': i
                    } for i in year_options],
                    value=latest_year),
                html.Label('Region: '),
                dcc.Dropdown(
                    id="Region",
                    options=[{
                        'label': i,
                        'value': i
                    } for i in region_options],
                    multi=True,
                    value=['(Total)']),
                html.Label('Sub-Region: '),
                dcc.Dropdown(
                    id="Sub_region",
                    options=[{
                        'label': i,
                        'value': i
                    } for i in sub_region_options],
                    multi=True,
                    value=['(Total)']
                    ),
                html.Label('Country: '),
                dcc.Dropdown(
                    id="Country",
                    options=[{
                        'label': i,
                        'value': i
                    } for i in country_options],
                    multi=True,
                    value=['(Total)']),
                html.Label('Vendor: '),
                dcc.Dropdown(
                    id="Vendor",
                    options=[{
                        'label': i,
                        'value': i
                    } for i in vendor_options],
                    value='(Total)'
                    ),
                html.Label('yaxis_type: '),
                dcc.RadioItems(
                    id='yaxis_type',
                    options=[{'label': i, 'value': i} for i in ['Units', 'Value ($)']],
                    value='Units',
                    labelStyle={'display': 'inline-block'}
                )
            ],
            style={'width': '18%','float': 'left','display': 'inline-block'}),
        html.Div([  
            dcc.Graph(id='main-graph')],
            style={'width': '80%', 'float': 'right', 'display': 'inline-block'})
        ])
])


# 更新区域选择框
@app.callback(
    dash.dependencies.Output('Region', 'options'),
    [dash.dependencies.Input('Year', 'value')])
def update_region_options(Year):
    region_options=['(Total)']+list(df.loc[df['Year']==Year,'Region'].unique())
    return [{'label': i, 'value': i} for i in region_options]



# 更新子区域选择框
@app.callback(
    dash.dependencies.Output('Sub_region', 'options'),
    [dash.dependencies.Input('Year', 'value'),
     dash.dependencies.Input('Region', 'value')])
def update_sub_region_options(Year,Region):
    if '(Total)' in Region or len(Region)==0:
        Region=df.loc[df['Year']==Year,'Region'].unique()
    sub_region_options=df.loc[(df['Year']==Year)&(df['Region'].isin(Region)),'Sub-region'].unique()
    #if len(sub_region_options)==0:
    #    sub_region_options=df.loc[df['Year']==Year,'Sub-region'].unique()
    sub_region_options=['(Total)']+list(sub_region_options)
    return [{'label': i, 'value': i} for i in sub_region_options]

# 更新国家选择框
@app.callback(
    dash.dependencies.Output('Country', 'options'),
    [dash.dependencies.Input('Year', 'value'),
     dash.dependencies.Input('Region', 'value'),
     dash.dependencies.Input('Sub_region', 'value')])
def update_country_options(Year,Region,Sub_region):
    if '(Total)' in Region  or len(Region)==0:
        Region=df.loc[df['Year']==Year,'Region'].unique()
    if '(Total)' in Sub_region or len(Sub_region)==0:
        Sub_region=df.loc[(df['Year']==Year)&(df['Region'].isin(Region)),'Sub-region'].unique()
    country_options=df.loc[df['Sub-region'].isin(Sub_region),'Country'].unique()
    country_options=['(Total)']+list(country_options)
    return [{'label': i, 'value': i} for i in country_options]


# 更新品牌选择框
@app.callback(
    dash.dependencies.Output('Vendor', 'options'),
    [dash.dependencies.Input('Year', 'value'),
     dash.dependencies.Input('Region', 'value'),
     dash.dependencies.Input('Sub_region', 'value'),
     dash.dependencies.Input('Country', 'value')])               
def update_vendor_options(Year,Region,Sub_region,Country):
    if '(Total)' in Region  or len(Region)==0:
        Region=df.loc[df['Year']==Year,'Region'].unique()
    if '(Total)' in Sub_region or len(Sub_region)==0:
        Sub_region=df.loc[(df['Year']==Year)&(df['Region'].isin(Region)),'Sub-region'].unique()
    if '(Total)' in Country or len(Country)==0:
        Country=df.loc[(df['Year']==Year)&(df['Sub-region'].isin(Sub_region)),'Country'].unique()
    vendor_options=df.loc[df['Country'].isin(Country),'Vendor'].unique()
    #if len(vendor_options)==0:
    #    vendor_options=df.loc[df['Year']==Year,'Vendor'].unique()
    vendor_options=['(Total)']+list(vendor_options)
    return [{'label': i, 'value': i} for i in vendor_options]



@app.callback(
    dash.dependencies.Output('main-graph', 'figure'),
    [dash.dependencies.Input('Year', 'value'),
     dash.dependencies.Input('Region', 'value'),
     dash.dependencies.Input('Sub_region', 'value'),
     dash.dependencies.Input('Country', 'value'),
     dash.dependencies.Input('Vendor', 'value'),                            
     dash.dependencies.Input('yaxis_type', 'value')])
def update_graph_main(Year,Region,Sub_region,Country,Vendor,yaxis_type):

    title_Region=''
    if '(Total)' in Region  or len(Region)==0:
        title_Region='All Regions'
        Region=df.loc[df['Year']==Year,'Region'].unique()
    title_Sub_region=''
    if '(Total)' in Sub_region or len(Sub_region)==0:
        title_Sub_region='All Sub-regions'
        Sub_region=df.loc[(df['Year']==Year)&(df['Region'].isin(Region)),'Sub-region'].unique()
    if '(Total)' in Country or len(Country)==0:
        title_Country = 'All Countrys' if title_Region == 'All Regions' and title_Sub_region=='All Sub-regions' else 'Some Countrys'
        Country=df.loc[(df['Year']==Year)&(df['Sub-region'].isin(Sub_region)),'Country'].unique()
    elif len(Country) >1:
        title_Country = 'Some Countrys'
    else:
        title_Country = Country[0]
    # 部分修正    
    if title_Country == 'Some Countrys' and len(Country)==1:
        title_Region = Country[0]
    
    df_plot = df[(df['Year']==Year)&(df['Country'].isin(Country))]
    if Vendor == '(Total)' or Vendor is None :
        pv = pd.pivot_table(
            df_plot,
            index=['Vendor'],
            columns=['Quarter'],
            values=[yaxis_type],
            aggfunc=np.sum,
            fill_value=0)
    else:
        pv = pd.pivot_table(
            df_plot[df_plot['Vendor']==Vendor],
            index=['Model'],
            columns=['Quarter'],
            values=[yaxis_type],
            aggfunc=np.sum,
            fill_value=0)
    quarter_options=[c[1] for c in pv.columns]
    pv[(yaxis_type,'total')]=pv.sum(axis=1)
    pv=pv.sort_values(by=[(yaxis_type,'total')],axis=0,ascending=False)#.iloc[:20,:]
    trace = []
    for i in quarter_options:
        trace.append(go.Bar(x=pv.index, y=pv[(yaxis_type,i)], name='Q%d'%i))


    title_Vendor = 'Total Vendors' if Vendor == '(Total)' else Vendor 
                                
        
    return {
        'data': trace,
        'layout':
        go.Layout(
            title='Mobile Sales ({}) of {} in {}'.format(yaxis_type,title_Vendor,title_Country),
            barmode='stack')
    }



if __name__ == '__main__':
    #app.run_server(debug=True)
	#app.run_server(debug=True,host='0.0.0.0')
	app.run(debug=True,host='0.0.0.0')
