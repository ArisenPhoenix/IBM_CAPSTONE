# Import required libraries
import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px

pd.set_option('display.max_columns', None)

# Read the spacex data into pandas dataframe
spacex_df = pd.read_csv("spacex_launch_dash.csv")
max_payload = spacex_df['Payload Mass (kg)'].max()
min_payload = spacex_df['Payload Mass (kg)'].min()
# Remove the additional wannabe index called Unnamed: 0
spacex_df.drop('Unnamed: 0', axis=1, inplace=True)
launch_sites = spacex_df['Launch Site'].unique()
launch_sites = [{'label': site, 'value': site} for site in launch_sites]
launch_sites.insert(0,{'label': 'All Sites', 'value': 'ALL'})

# print(spacex_df.groupby(by="class").count())
test = spacex_df.groupby(by="Launch Site")['class'].count()
print("TEST \n ", test)
# Create a dash application
app = dash.Dash(__name__)

# Create an app layout
app.layout = html.Div(children=[html.H1('SpaceX Launch Records Dashboard',
                                        style={'textAlign': 'center', 'color': '#503D36',
                                               'font-size': 40}),
                                # TASK 1: Add a dropdown list to enable Launch Site selection
                                dcc.Dropdown(id='site-dropdown',
                                             options=launch_sites,
                                             placeholder="Select a Launch Site Here",
                                             searchable=True,
                                             value='ALL',
                                             ),
                                # The default select value is for ALL sites
                                html.Br(),
                                # TASK 2: Add a pie chart to show the total successful launches count for all sites
                                # If a specific launch site was selected, show the Success vs. Failed counts for the site
                                html.P("Launch Success Pie Chart"),
                                html.Div([], id='success-pie-chart'),
                                html.Br(),
                                html.P("Payload range (Kg):"),
                                # TASK 3: Add a slider to select payload range
                                dcc.RangeSlider(id='payload-slider',
                                                min=0, max=10000, step=1000,
                                                marks={0: '0',
                                                       100: '100'},
                                                value=[min_payload, max_payload]),

                                # TASK 4: Add a scatter chart to show the correlation between payload and launch success
                                html.Div([], id='success-payload-scatter-chart'),
                                ])


# TASK 2:
# Add a callback function for `site-dropdown` as input, `success-pie-chart` as output
@app.callback([Output(component_id='success-pie-chart', component_property='children')],
              Input(component_id='site-dropdown', component_property='value')
              )
# Place to define the callback function .
# TASK 3F
def get_pie_chart(value):
    if value == "ALL":
        filtered_df = spacex_df.groupby(by="Launch Site")['class'].count()
        pie_fig = px.pie(filtered_df, values=filtered_df.values, names=filtered_df.index, title="Pie Chart")
        
    else:
        filtered_df = spacex_df[spacex_df['Launch Site'] == value].groupby('class').count()
        filtered_df.index = ['Success' if index == 1 else 'Failure' for index in filtered_df.index]
        filtered_df.sort_index(ascending=False)
        values = [filtered_df.values[0][0], filtered_df.values[1][0]]
        pie_fig = px.pie(values=values, names=filtered_df.index, title="Pie Chart")
    return [dcc.Graph(figure=pie_fig)]


# TASK 4:
# Add a callback function for `site-dropdown` and `payload-slider` as inputs, `success-payload-scatter-chart` as output
@app.callback(Output(component_id='success-payload-scatter-chart', component_property='children'),
              [Input(component_id='site-dropdown', component_property='value'),
               Input(component_id="payload-slider", component_property="value")]
              )
def update_slider_and_input(value, slider):
    if value == "ALL":
        typed_df = spacex_df[spacex_df['Payload Mass (kg)'] > slider[0]]
        typed_df = typed_df[typed_df['Payload Mass (kg)'] < slider[1]]
        scatter_fig = px.scatter(typed_df, x='Payload Mass (kg)', y='class', color="Booster Version")
    else:
        typed_df = spacex_df[spacex_df["Launch Site"] == value]
        typed_df = typed_df[typed_df['Payload Mass (kg)'] > slider[0]]
        typed_df = typed_df[typed_df['Payload Mass (kg)'] < slider[1]]
        scatter_fig = px.scatter(typed_df, x='Payload Mass (kg)', y='class', color="Booster Version")
    return [dcc.Graph(figure=scatter_fig)]

# Run the app
if __name__ == '__main__':
    app.run()
