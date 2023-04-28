import pandas as pd

import plotly.graph_objs as go
import os
from dash import Dash, dcc, html

#####################################################################################
traindata = pd.read_csv(r"dataset\traindataSingle.csv")
single_df = pd.read_csv(r"dataset\outputSingle.csv")
multi_df = pd.read_csv(r"dataset\outputMulti.csv")
#####################################################################################
font_family = "Arial"
#####################################################################################
fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=single_df["Date"],
                         y=single_df["Actual_Open"],
                         name="Actual",
                         mode="lines",
                         line = dict(color="rgb(0,0,0)")
                        )
             )
fig3.add_trace(go.Scatter(x=single_df["Date"],
                         y=single_df["Predicted_Open"],
                         name="Forecast (Single LSTM)",
                         mode="lines",
                         line = dict(color="rgb(0,136,204)")
                        )
             )
fig3.add_trace(go.Scatter(x=multi_df["Date"],
                          y=multi_df["Predicted_Open"],
                          name="Forecast (Multi LSTM)",
                          mode="lines",
                          line=dict(color="rgb(250,128,114)")
                          )
               )
# Figure Customize
fig3.update_traces(hovertemplate = "<br><i>Open Price</i>: %{y}",
                 )

fig3.update_layout(
                title=dict(text="AAPL Stock Daily Open Price Forecast (2023)",
                             font = dict(size=14, family=font_family),
                             xanchor = "left",
                             pad_l=150
                            ),
                  legend=dict(
                      # title = dict(text = "Open Price" ,
                      #                      font =dict(size=12, family=font_family)),
                              bgcolor = "rgb(238,238,238)",
                              bordercolor = "rgb(0,0,0)",
                              font =dict(size=12, family=font_family),
                              yanchor="top",
                              y=1.15,
                              xanchor="left",
                              x=0.01,
                              orientation="h",
                             ),
                  plot_bgcolor='rgb(255,255,255)',
                  hovermode="x unified",
                  width=600,
                  height=400,
                 )
fig3.update_xaxes(
                 title=dict(text = "Date", font=dict(size= 12, family=font_family)),
                 minor = dict(griddash="dashdot",
                              showgrid=True,
                              gridwidth = 1,
                              gridcolor= "rgb(151,151,151)",
                             ),
                 gridcolor = "rgb(151,151,151)",
                 showgrid=True,
                )
fig3.update_yaxes(title=dict(text = "Open Price", font=dict(size= 12, family=font_family)),
                gridcolor="rgb(151,151,151)",
                showgrid=True,
                )

## Save a screenshot of the plot
if not os.path.exists("images"):
    os.mkdir("images")
fig3.write_image(r"images\mainplot.png")


#####################################################################################
app = Dash()
# 1 dash app
app.layout = html.Div([
    html.Div([
        dcc.Graph(figure=fig3),
                  ]),
])



# Run the app
if __name__ == '__main__':
    app.run_server(debug=False, port=3002)


