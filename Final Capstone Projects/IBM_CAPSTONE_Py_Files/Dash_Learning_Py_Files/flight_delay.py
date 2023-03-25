# Import required libraries
import pandas as pd
import plotly.graph_objects as go
import dash
from dash import html
from dash import dcc
from dash.dependencies import Input, Output
import plotly.express as px

# Read the airline data into pandas dataframe
airline_data =  pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/airline_data.csv',
                            encoding = "ISO-8859-1",
                            dtype={'Div1Airport': str, 'Div1TailNum': str,
                                   'Div2Airport': str, 'Div2TailNum': str})

# Create a dash application
app = dash.Dash(__name__)

# Build dash app layout
app.layout = html.Div(
    children=[html.H1(),
        html.Div(["Input Year: ", dcc.Input()],
                 style={'font-size': 30}),
              
        html.Br(),
        html.Br(),
        
        html.Div([
            html.Div(),
            html.Div()
        ], style={'display': 'flex'}),

        html.Div([
            html.Div(),
            html.Div()
        ], style={'display': 'flex'}),

        html.Div(style = {'width': '65%'})
])