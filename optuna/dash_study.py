import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import flask
import joblib
import optuna
import datetime

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def get_layout():
    try:
        study = joblib.load('study.pkl')
    except FileNotFoundError:
        return html.Div(children=[dcc.Markdown(children='''
        ### Optuna Study Dashboard
        >
        > **Found no study.pkl file**.
        >

        As soon as a pickled study file is available, you can refresh the page to load the dashboard.

        - If an Optuna study is running, please wait for the first trial to finish.
        - If no study is running, please copy a pickled study file in the folder or run a study with: 
            `python optuna_training.py`.
        ''')])

    n_trials = len(study.trials)
    avg_trial_time = datetime.timedelta(seconds=sum([t.duration.seconds for t in study.trials]) / n_trials)

    summary_text = 'Presenting {} trials of average run time {}'.format(n_trials, avg_trial_time)

    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    n_pruned_trials = len(pruned_trials)
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    n_completed_trials = len(completed_trials)

    trial_param_names, best_values = [k for k, i in study.best_trial.params.items()], []
    for _, item in study.best_trial.params.items():
        if isinstance(item, float):
            best_values.append('{:.2e}'.format(item))
        else:
            best_values.append(item)

    summary_fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'domain'}, {'type': 'domain'}]])
    summary_fig.add_trace(go.Table(header=dict(values=['Parameter Name', 'Best Value']),
                                   cells=dict(values=[trial_param_names, best_values])), 1, 1)
    summary_fig.add_trace(go.Pie(
        labels=['Completed', 'Pruned'],
        values=[n_completed_trials, n_pruned_trials],
        hole=.3, pull=[0, 0.2]), 1, 2)
    summary_fig.update_layout(title_text='Study Summary')

    trials_dataframe = study.trials_dataframe()

    children = [

        html.H1(children='Optuna Study Dashboard'),
        html.Div(children=summary_text),
        dcc.Graph(figure=summary_fig),
        dcc.Graph(id='Intermediate Values', figure=optuna.visualization.plot_intermediate_values(study)),
        dcc.Graph(id='Contour', figure=optuna.visualization.plot_contour(study)),
        dcc.Graph(id='Parallel Coordinate', figure=optuna.visualization.plot_parallel_coordinate(study)),
        dcc.Graph(id='Optimisation History', figure=optuna.visualization.plot_optimization_history(study)),
        dcc.Graph(id='Slice', figure=optuna.visualization.plot_slice(study)),
        dash_table.DataTable(id='Trials Table', columns=[{"name": i, "id": i} for i in trials_dataframe.columns],
                             data=trials_dataframe.to_dict('records'))
    ]

    try:
        children.append(dcc.Graph(id='Hyperparameter Importance',
                                  figure=optuna.visualization.plot_param_importances(study)))
    except AssertionError:  # in some cases, the hyperparmeter importance plot will fail with this error
        pass

    layout = html.Div(children)
    return layout


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
server = flask.Flask(__name__)
app = dash.Dash(__name__, server=server, external_stylesheets=external_stylesheets)
app.config.suppress_callback_exceptions = True
app.layout = get_layout

if __name__ == '__main__':
    app.run_server(host='0.0.0.0', debug=False, port=8050)

