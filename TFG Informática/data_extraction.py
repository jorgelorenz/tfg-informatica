#En este fichero saco datos y gráficas de retornos en las ventanas que quiero estudiar

###Lista stocks:
### Compañías
### Índices

###Por cada stock coger varias ventanas temporales, escenarios disintos podemos basarnos en https://www.sas.upenn.edu/~fdiebold/papers/misc/Brownlees.pdf

### Para elegir ventanas temporales traemos los datos y graficamos para elegir escenarios normales y de estrés. 

# def get_arch_model(ticker, start_date, end_date):
#     sp500 = yf.download(ticker, start=start_date, end=end_date)
#     returns = 100 * sp500['Close'].pct_change().dropna()
#     am = arch_model(returns)

#     res = am.fit()

#     print(res.summary())

import datetime as dt
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
# from arch import arch_model
# from fpdf import FPDF
from PIL import Image
from dateutil.parser import parse
import numpy as np


start_date = dt.datetime(2000, 1, 1)
end_date = dt.datetime(2024, 12, 31)

def download_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    data.columns = data.columns.droplevel(1)
    data['Daily Return'] = data['Close'].pct_change()
    data['Daily log return'] = np.log(1 + data['Close'].pct_change().dropna())
    filename = 'datasets/' + ticker + '_' + start_date.strftime("%Y") + '_' + end_date.strftime("%Y") + '.csv'
    
    data.to_csv( filename, index_label=False)
    
def plot_daily_returns(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
    except:
        return
    if data.empty:
        return
    data = yf.download(ticker, start=start_date, end=end_date)
    data['Daily Return'] = data['Close'].pct_change()
    data = data.dropna()

    fig = px.line(
        data,
        x=data.index,
        y='Daily Return',
        title=f'Rendimientos Diarios de {ticker} ({start_date} a {end_date})',
        labels={
            'x': 'Fecha',
            'Daily Return': 'Rendimiento Diario'
        }
    )

def plot_daily_returns(ticker, start_date, end_date, sector='', study_windows=None):
    data = yf.download(ticker, start=start_date, end=end_date)
    data['Daily Return'] = data['Close'].pct_change()
    data = data.dropna()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=data.index, 
        y=data['Daily Return'], 
        mode='lines',
        name='Retorno Diario'
    ))

    if study_windows != None:
        for window in study_windows:
            fig.add_shape(
                type="rect",
                x0=window['start'],
                x1=window['end'],
                y0=min(data['Daily Return']),
                y1=max(data['Daily Return']),
                opacity=0.7,
                layer="below",
                line=dict(
                    width=3,
                    color="red"
                )
            )

    fig.update_layout(
        title=f'Rendimientos Diarios de {ticker} ({start_date} a {end_date})',
        xaxis_title='Fecha',
        yaxis_title='Retorno Diario',
        template='plotly_white'
    )


    output_dir = "images/returns"
    os.makedirs(output_dir, exist_ok=True)

    if sector!='':
        output_dir += f'/{sector}'
        os.makedirs(output_dir, exist_ok=True)
        
    output_path = os.path.join(output_dir, f"{ticker}_returns_2000_to_2024.png")
    fig.write_image(output_path)

    print(f"Gráfico guardado en: {output_path}")
    
def get_sp500_tickers_by_sector():
    """
    Obtiene los tickers de las compañías del S&P 500 agrupados por sector.

    :return: Diccionario con sectores como claves y listas de tickers como valores.
    """
    # URL oficial con la lista del S&P 500 desde Wikipedia
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    
    # Descargar tabla desde Wikipedia
    tables = pd.read_html(url)
    sp500_table = tables[0]
    
    # Agrupar por sector
    tickers_by_sector = sp500_table.groupby('GICS Sector')['Symbol'].apply(list).to_dict()
    
    return tickers_by_sector

def get_major_indices_tickers():
    """
    Devuelve tickers de índices bursátiles importantes.

    :return: Diccionario con nombres de índices y sus tickers.
    """
    indices = {
        "S&P 500": "^GSPC",
        # "Dow Jones Industrial Average": "^DJI",
        # "NASDAQ 100": "^NDX",
        # "Russell 2000": "^RUT",
        # "FTSE 100": "^FTSE",
        # "DAX": "^GDAXI",
        # "Nikkei 225": "^N225",
        # "Hang Seng": "^HSI",
        # "Shanghai Composite": "000001.SS",
        # "MSCI World": "MSCI",
        "IBEX 35": "^IBEX"
    }
    return indices

if __name__ == "__main__":
    #Parámetros de uso
    all = True
    
    study_windows = [
            {'start': '2018-01-01', 'end': '2021-12-31'},
            {'start': '2021-01-01', 'end': '2024-12-31'}
        ]
    
    # tickers_by_sector = get_sp500_tickers_by_sector()
    
    indices_tickers = get_major_indices_tickers()
    tickers_by_sector = dict()
    
    #Dibujar gráficas de retornos
    for index, ticker in indices_tickers.items():
        print(index)
        plot_daily_returns(ticker, start_date, end_date, study_windows=study_windows)
        if not all:
            break
    
    for index, tickers in tickers_by_sector.items():
        for ticker in tickers[:2]:
            if ticker=='BF.B':
                continue
            print(ticker)
            plot_daily_returns(ticker, start_date, end_date, sector=index, study_windows=study_windows)
        if not all:
            break
    
    
    #Descargar datos a csv de las ventanas que queramos para tenerlo guardados
        
    for index, ticker in indices_tickers.items():
        print(index)
        for window in study_windows:
            start_date = parse(window['start'])
            end_date= parse(window['end'])
            download_data(ticker, start_date, end_date)
            
    for index, tickers in tickers_by_sector.items():
        for ticker in tickers[:2]:
            print(ticker)
            for window in study_windows:
                start_date = parse(window['start'])
                end_date= parse(window['end'])
                download_data(ticker, start_date, end_date)
            