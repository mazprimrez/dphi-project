import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output

import pandas as pd
import numpy as np

from pycaret.classification import *

#load model
model = load_model('deployment_28122020')

#load the data
loan_data  = pd.read_csv("https://raw.githubusercontent.com/dphi-official/Datasets/master/Loan_Data/loan_train.csv" )

#Dash
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

app.layout = html.Div([
    #Baris 1
    dbc.Row([
            html.H1("Loan Prediction"),
            html.Br()
    ], justify='center'),

    #Baris 2
    dbc.Row([
        dbc.Col([

            html.H5('Married: '),
            dcc.Dropdown(id='married', value='No',
                        options = [{'label': 'No', 'value': 'No'},
                                   {'label': 'Yes', 'value': 'Yes'}],
                        ),
            html.H5('Dependents: '),
            dcc.Dropdown(id='dependents', value='1',
                        options = [{'label': '1', 'value': '1'},
                                   {'label': '2', 'value': '2'},
                                   {'label': '3+', 'value': '3+'}],
                        ),
            html.H5('Education: '),
            dcc.Dropdown(id='education', value='Graduate',
                        options = [{'label': 'Graduate', 'value': 'Graduate'},
                                   {'label': 'Not Graduate', 'value': 'Not Graduate'}],
                        ),
            html.H5('Self-Employed: '),
            dcc.Dropdown(id='employed', value='Yes',
                        options = [{'label': 'Yes', 'value': 'Yes'},
                                   {'label': 'No', 'value': 'No'}],
                        ),
            html.H5('LoanAmount: '),
            dcc.Input(id="loan", type="text", placeholder=""),
        ], xs=6, sm=6, md=4, lg=2, xl=2),

        dbc.Col([
            html.H5('Gender:'),
            dcc.Dropdown(id='gender', value='Male',
                        options = [{'label': 'Male', 'value': 'Male'},
                                   {'label': 'Female', 'value': 'Female'}],
                        ),
            html.H5('ApplicantIncome:'),
            dcc.Input(id="app", type="text", placeholder=""),
            html.H5('CoapplicantIncome:'),
            dcc.Input(id="coapp", type="text", placeholder=""),
            html.H5('Loan_Amount_Term:'),
            dcc.Input(id="term", type="text", placeholder=""),
            html.H5('Credit_History:'),
            dcc.Dropdown(id='credit', value=1,
                        options = [{'label': '1', 'value': 1},
                                   {'label': '0', 'value': 0}],
                        ),
            html.H5('Property_Area:'),
            dcc.Dropdown(id='area', value='Semiurban',
                        options = [{'label': 'Semiurban', 'value': 'Semiurban'},
                                   {'label': 'Urban', 'value': 'Urban'},
                                   {'label': 'Rural', 'value': 'Rural'}],
                        ),

        ],xs=6, sm=6, md=4, lg=2, xl=2)

    ], justify = 'center'),

    #Baris 3
    dbc.Row([
        dbc.Col([
            html.Center(html.H5('Loan Status:')),
            html.Center(html.H3(id='result2')),
        ],xs=5, sm=5, md=5, lg=5, xl=5),
    ], justify = 'center')
])

@app.callback(
    Output('result2', 'children'),
    [Input('gender','value'),
    Input('married','value'),
    Input('dependents','value'),
    Input('education','value'),
    Input('employed','value'),
    Input('app','value'),
    Input('coapp','value'),
    Input('loan','value'),
    Input('term','value'),
    Input('credit','value'),
    Input('area','value')
    ],
)

def label(gender=np.nan,married=np.nan,dependents=np.nan,education=np.nan,employed=np.nan,app=np.nan,coapp=np.nan,loan=np.nan,
                 term=np.nan,credit=np.nan,area=np.nan):

    x = pd.DataFrame({loan_data.columns[0]:0,
              loan_data.columns[1]:'',
              loan_data.columns[2]:gender,
              loan_data.columns[3]:married,
              loan_data.columns[4]:dependents,
              loan_data.columns[5]:education,
              loan_data.columns[6]:employed,
              loan_data.columns[7]:app,
              loan_data.columns[8]:coapp,
              loan_data.columns[9]:loan,
              loan_data.columns[10]:term,
              loan_data.columns[11]:credit,
              loan_data.columns[12]:area,
             }, index=[0])
    pred = predict_model(model, data=x)

    if pred.Label[0]==1:
        return 'Approved'
    else:
        return 'Reject'

if __name__ == "__main__":
    app.run_server(debug=True, use_reloader=False)