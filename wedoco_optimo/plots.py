from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio

def plot_from_def(plot_def, df, show=True, save_to_file=False, filename='plot.html'):
    fig = make_subplots(rows=len(plot_def)-1, cols=1, shared_xaxes=True, vertical_spacing=0.02)
    for i, p in enumerate(p for p in plot_def.keys() if p != 'x'):
        fig.update_yaxes(title_text=p, row=i+1, col=1)
        offset = plot_def[p]['offset'] if 'offset' in plot_def[p].keys() is not None else 0
        factor = plot_def[p]['factor'] if 'factor' in plot_def[p].keys() is not None else 1
        for v in plot_def[p]['vars']:
            fig.add_trace(go.Scatter(x=plot_def['x']['values'], name=v,
                                     y=df[v]*factor+offset), row=i+1, col=1)
            
    # Show x-axis title and ticks only on the last subplot
    fig.update_xaxes(title_text=plot_def['x']['title'], row=i+1, col=1, showticklabels=True)
    # Update layout to show a shared vertical line across all subplots when hovering
    fig.update_layout(hovermode="x unified", hoversubplots='axis')
    
    fig.update_layout(height=800)
    if show: 
        fig.show()
    if save_to_file:
        pio.write_html(fig, file=filename, auto_open=False)
