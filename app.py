import dash
import dash_core_components as dcc
import dash_html_components as html

import plotly.graph_objects as go
import numpy as np
import plotly.express as px
import pandas as pd

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


# fig 0

bigram_df = pd.read_csv('data/bigram_df.csv')

fig0 = px.bar(bigram_df, y="bigram", x="freq", color = 'freq', animation_frame="year", orientation = 'h',

             labels={ # replaces default labels by column name
                "bigram": "Two-Word Phrases",  "freq": "Frequency"},
             color_continuous_scale=px.colors.sequential.Viridis,
             template="simple_white")

fig0.update_layout(yaxis={'categoryorder':'total ascending'})

annotations = []

# Title
annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,
                              xanchor='left', yanchor='bottom',
                              text="Key Phrases of the Year",
                              font=dict(family='Arial',
                                        size=24,
                                        color='rgb(37,37,37)'),
                              showarrow=False))

fig0.update_layout(annotations=annotations)




# fig 1

us_df = pd.read_csv("data/us_df.csv")

title = 'Main Source for News'
labels = ['U.S.']
colors = ['rgb(49,130,189)']

mode_size = [8, 8, 12, 8]
line_size = [2, 2, 4, 2]

x_data = np.vstack((np.arange(1999, 2014),)*1)

y_data = np.array([np.array(us_df.us_index)])

fig1 = go.Figure()

for i in range(0, 1):
    fig1.add_trace(go.Scatter(x=x_data[i], y=y_data[i], mode='lines',
        name=labels[i],
        line=dict(color=colors[i], width=line_size[i]),
        connectgaps=True,
    ))

    # endpoints
    fig1.add_trace(go.Scatter(
        x=[x_data[i][0], x_data[i][-1]],
        y=[y_data[i][0], y_data[i][-1]],
        mode='markers',
        marker=dict(color=colors[i], size=mode_size[i])
    ))

fig1.update_layout(
    xaxis=dict(
        showline=True,
        showgrid=False,
        showticklabels=True,
        linecolor='rgb(204, 204, 204)',
        linewidth=2,
        ticks='outside',
        tickfont=dict(
            family='Arial',
            size=12,
            color='rgb(82, 82, 82)',
        ),
    ),
    yaxis=dict(
        showgrid=False,
        zeroline=False,
        showline=False,
        showticklabels=False,
    ),
    autosize=False,
    margin=dict(
        autoexpand=True,
        l=120,
        r=20,
        t=110,
    ),
    showlegend=False,
    plot_bgcolor='white'
)

annotations = []

# Adding labels
for y_trace, label, color in zip(y_data, labels, colors):
    # labeling the left_side of the plot
    annotations.append(dict(xref='paper', x=0.05, y=y_trace[0],
                                  xanchor='right', yanchor='middle',
                                  text=label + ' {}%'.format(y_trace[0]),
                                  font=dict(family='Arial',
                                            size=13),
                                  showarrow=False))
    # labeling the right_side of the plot
    annotations.append(dict(xref='paper', x=0.95, y=y_trace[11],
                                  xanchor='left', yanchor='middle',
                                  text='{}%'.format(y_trace[11]),
                                  font=dict(family='Arial',
                                            size=13),
                                  showarrow=False))
# Title
annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,
                              xanchor='left', yanchor='bottom',
                              text='\"United States\" Appearance',
                              font=dict(family='Arial',
                                        size=24,
                                        color='rgb(37,37,37)'),
                              showarrow=False))


fig1.update_layout(annotations=annotations)

# fig2

# polarity scatter plot


fig2 = go.Figure()

df_pol = pd.read_csv('data/df_pol.csv')


# make subplots
fig2.add_trace(
    go.Scatter(
        x=df_pol[df_pol['polarity'] >= 0]['year'],
        y=df_pol[df_pol['polarity'] >= 0]['polarity'],
        mode="markers",
        marker=dict(color='red')
    ))

fig2.add_trace(
    go.Scatter(
        x=df_pol[df_pol['polarity'] < 0]['year'],
        y=df_pol[df_pol['polarity'] < 0]['polarity'],
        mode="markers",
        marker=dict(color='blue')
    ))


fig2.update_layout(
    autosize=False,
    showlegend=False,
    plot_bgcolor='white'
)



annotations = []

# Title
annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,
                              xanchor='left', yanchor='bottom',
                              text='Style: Positive vs Negative',
                              font=dict(family='Arial',
                                        size=24,
                                        color='rgb(37,37,37)'),
                              showarrow=False))

fig2.update_layout(annotations=annotations)

# fig3

df_pro = pd.read_csv('data/df_pro.csv')

fig3 = px.histogram(df_pro, x='polarity', histnorm='probability',
                facet_col='producer', facet_col_wrap = 4)

fig3.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))


fig3.update_layout(
    title="Producer Style Comparison",
    font=dict(
        family="Arial"
    )
)

fig3.update_xaxes(title_text='')
fig3.update_yaxes(title_text='')


app.layout = html.Div(children=[
    html.H1(children='CBS 60 Minutes Analysis'),

    html.Div(children='''
        CBS 60 Minutes offers access to transcripts of nearly 500 hours of video from 1997 - 2014.

        Available in the US Newsstream on ProQuest Database.


    '''),


    html.Div(dcc.Graph(
        id='example-graph',
        figure=fig0
    )),

    html.Div(dcc.Graph(
        id='example-graph1',
        figure=fig1)),

    html.Div(dcc.Graph(
        id='example-graph2',
        figure=fig2)),

    html.Div(dcc.Graph(
        id='example-graph3',
        figure=fig3))


        ])

if __name__ == '__main__':
    app.run_server(debug=True)
