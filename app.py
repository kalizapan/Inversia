# app.py
import os
import time
import random
from flask import url_for
import re
import requests
from requests import RequestException
from functools import lru_cache
import numpy as np
import pandas as pd
from werkzeug.security import generate_password_hash, check_password_hash
from functools import lru_cache
from cachetools import TTLCache, cached
import yfinance as yf
import requests
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import json
from datetime import datetime, timedelta

# Matplotlib sin GUI
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ReportLab Platypus para PDF estilizado
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
    Table, TableStyle, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch

# Envío de email vía SMTP
import smtplib
from email.message import EmailMessage
from requests import RequestException

# Flask & SQLAlchemy
from flask import Flask, render_template, request, redirect, session, url_for
from sqlalchemy import (
    create_engine, Column, Integer, String, ForeignKey,
    UniqueConstraint
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

# — rutas absolutas —
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR   = os.path.join(BASE_DIR, 'static')
TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')

app = Flask(
    __name__,
    static_folder=STATIC_DIR,
    template_folder=TEMPLATE_DIR
)
app.secret_key = os.environ.get('SECRET_KEY', os.urandom(24).hex())

# -Configuracion Cache-
td_cache = TTLCache(maxsize=100, ttl=43200)

# — Credenciales de correo SMTP —
EMAIL_USER = os.environ.get('EMAIL_USER', 'inversiacontact@gmail.com')
EMAIL_PASS = os.environ.get('EMAIL_PASS', 'ovgu mmmo dakz sfnh')

# — Twelve Data API config —
TD_API_KEY = '3a14abf485024ff8874242de3c165e55'
TD_URL     = 'https://api.twelvedata.com/time_series'
TD_STOCKS_URL = 'https://api.twelvedata.com/stocks'

# - OpenRouter Config -
OPENROUTER_URL     = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = "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"

# — Máximos por campo para validación de longitud —
MAX_LEN = {
    'email':       50,
    'password':    64,
    'first_name':  30,
    'last_name':   30,
    'institucion': 50,
    'ticker':      5,
}

# — ORM & Base de datos —
engine       = create_engine('sqlite:///database.db', echo=False)
Base         = declarative_base()
SessionLocal = sessionmaker(bind=engine)

class User(Base):
    __tablename__ = 'users'
    id           = Column(Integer, primary_key=True, autoincrement=True)
    email        = Column(String, unique=True, nullable=False)
    password     = Column(String, nullable=False)
    first_name   = Column(String, nullable=False)
    last_name    = Column(String, nullable=False)
    motivo       = Column(String, nullable=False)
    institucion  = Column(String, nullable=False)
    portfolio    = relationship('PortfolioItem', back_populates='user')

class PortfolioItem(Base):
    __tablename__ = 'portfolio_items'
    __table_args__ = (
        UniqueConstraint('user_id', 'ticker', name='uix_user_ticker'),
    )
    id      = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    ticker  = Column(String, nullable=False)
    user    = relationship('User', back_populates='portfolio')

class TickerHistory(Base):
    __tablename__ = 'ticker_history'
    id        = Column(Integer, primary_key=True)
    user_id   = Column(Integer, ForeignKey('users.id'), nullable=False)
    ticker    = Column(String, nullable=False)
    timestamp = Column(String, default=lambda: time.strftime('%Y-%m-%d %H:%M:%S'))

    user = relationship('User')

class PasswordReset(Base):
    __tablename__ = 'password_resets'
    id      = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    code    = Column(String(6), nullable=False)
    user    = relationship('User')

Base.metadata.create_all(engine)

# Usuario de prueba
with SessionLocal() as db:
    if db.query(User).count() == 0:
        db.add(User(
            email='user@test.com',
            password=generate_password_hash('Password1!'),
            first_name='Usuario',
            last_name='Prueba',
            motivo='Educativo',
            institucion='Universidad X'
        ))
        db.commit()

@cached(td_cache)
def get_td_timeseries(symbol: str,
                      interval: str = '1day',
                      outputsize: int = 100,
                      retries: int = 3,
                      wait_seconds: int = 10) -> dict:
    """
    Llama a TwelveData controlando errores de cuota por minuto.
    """
    params = {
        'symbol':     symbol,
        'interval':   interval,
        'outputsize': outputsize,
        'apikey':     TD_API_KEY,
        'format':     'JSON'
    }

    for attempt in range(retries):
        try:
            resp = requests.get(TD_URL, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            if data.get('status') == 'error':
                msg = data.get('message', 'Error desconocido')
                if "api credits" in msg.lower():
                    print(f"[LÍMITE] Esperando {wait_seconds} segundos por créditos...")
                    time.sleep(wait_seconds)
                    continue
                raise ValueError(msg)

            return data

        except RequestException as e:
            print(f"[ERROR] Intento {attempt+1} fallido para {symbol}: {e}")
            if attempt < retries - 1:
                time.sleep(wait_seconds)
            else:
                raise

# Tipo de cambio desde frankfurter
def get_exchange_rates(base='USD'):
    try:
        url = f'https://api.frankfurter.app/latest?from={base}&to=MXN,EUR,GBP,JPY'
        r = requests.get(url)
        data = r.json()
        return data['rates']
    except Exception as e:
        print(f"[ERROR Frankfurter] {e}")
        return {'MXN': 0, 'EUR': 0, 'GBP': 0, 'JPY': 0}


from functools import lru_cache
import time
import requests

@lru_cache(maxsize=100)
def get_ticker_price(ticker):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            params = {
                'symbol': ticker,
                'interval': '1day',
                'outputsize': 1,
                'apikey': TD_API_KEY,
                'format': 'JSON'
            }
            resp = requests.get(TD_URL, params=params, timeout=20)
            data = resp.json()

            # Verifica errores de la API
            if data.get('status') == 'error':
                mensaje = data.get('message', 'Error desconocido')
                if "run out of api credits" in mensaje.lower():
                    print("[LÍMITE] Límite de API alcanzado. Esperando 60 segundos...")
                    time.sleep(60)
                    continue  # Reintenta después de esperar
                raise ValueError(mensaje)

            values = data.get('values')
            if not values or not isinstance(values, list):
                raise ValueError("Respuesta sin datos válidos")

            close_price = float(values[0]['close'])
            if close_price <= 0:
                raise ValueError("Precio inválido")

            return close_price

        except Exception as e:
            print(f"[ERROR] Fallo en intento {attempt+1} para {ticker}: {e}")
            if attempt < max_retries - 1:
                time.sleep(3)
            else:
                return None


def fetch_and_plot_td(ticker):
    """
    Consulta la API de TwelveData y dibuja el gráfico.
    Lanza ValueError con mensaje claro en caso de fallo.
    """
    params = {
        'symbol':     ticker,
        'interval':   '1day',
        'outputsize': 100,
        'apikey':     TD_API_KEY,
        'format':     'JSON'
    }
    try:
        resp = requests.get(TD_URL, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except RequestException as e:
        raise ValueError(f"Error de red al consultar la API: {e}")
    if data.get('status') == 'error':
        raise ValueError(f"Error de la API: {data.get('message','desconocido')}")
    values = data.get('values')
    if not isinstance(values, list) or not values:
        raise ValueError("No se recibieron datos de la API para el ticker.")
    df = pd.DataFrame(values)
    df['close'] = df['close'].astype(float)
    df['date']  = pd.to_datetime(df['datetime'])
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)
    df['Rendimiento'] = np.log(df['close'] / df['close'].shift(1))
    df.dropna(inplace=True)
    plt.figure(figsize=(10,5))
    df['Rendimiento'].plot(title=f'Rendimiento diario de {ticker}')
    plt.xlabel('Fecha'); plt.ylabel('Rt')
    os.makedirs(STATIC_DIR, exist_ok=True)
    img = f'{ticker}_rend_{int(time.time())}.png'
    plt.savefig(os.path.join(STATIC_DIR, img))
    plt.close()
    return img

def fetch_and_plot_td_plotly(ticker: str) -> str:
    # 1) usar la función cacheada
    data = get_td_timeseries(ticker, interval='1day', outputsize=100)
    values = data.get('values', [])
    if not values:
        raise ValueError(f"No hay datos para {ticker}")

    # 2) resto idéntico: DataFrame, fechas y cálculo de log-retornos
    df = pd.DataFrame(values)
    df['close'] = df['close'].astype(float)
    df['date']  = pd.to_datetime(df['datetime'])
    df.sort_values('date', inplace=True)

    df['log_close']   = np.log(df['close'])
    df['Rendimiento'] = df['log_close'].diff().dropna().astype(float)

    # DEBUG opcional:
    print(f"[DEBUG] {ticker} Rend min/max:",
          df['Rendimiento'].min(), df['Rendimiento'].max())

    # 3) crear la figura
    trace = go.Scatter(
        x=df['date'].tolist(),
        y=df['Rendimiento'].tolist(),
        mode='lines',
        name=f'Rend diario {ticker}'
    )
    layout = go.Layout(
        title=f'Rend diario de {ticker}',
        xaxis={'title':'Fecha'},
        yaxis={'title':'Rt'},
        template='plotly_white'
    )
    fig = go.Figure(data=[trace], layout=layout)

    # 4) serializar
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def monte_carlo_simulation_yf(ticker: str,
                              days: int = 30,
                              num_simulations: int = 100):
    """
    Simulación Monte Carlo con yfinance:
      - descarga cierres diarios
      - calcula log-retornos
      - genera num_simulations trayectorias de longitud days
      - guarda gráfico en STATIC_DIR
    Retorna: (filename, expected, p5, p95)
    Lanza ValueError si algo falla.
    """
    np.random.seed(42)
    # 1) Descargar históricos
    df = yf.download(ticker, period=f"{days*2}d", interval="1d", progress=False)
    if df.empty or 'Close' not in df:
        raise ValueError(f"No se obtuvieron datos para '{ticker}'")
    closes = df['Close'].dropna()
    if len(closes) < 2:
        raise ValueError(f"Datos insuficientes ({len(closes)} cierres) para '{ticker}'")

    # 2) Calcular log-retornos
    log_rets   = np.log(closes / closes.shift(1)).dropna()
    last_price = float(closes.iloc[-1])
    mu    = float(log_rets.mean())
    sigma = float(log_rets.std())

    # 3) Generar simulaciones como lista de listas
    sim_paths = []
    for _ in range(num_simulations):
        path = [last_price]
        for _ in range(1, days):
            shock = np.random.normal(mu - 0.5 * sigma**2, sigma)
            path.append(path[-1] * np.exp(shock))
        sim_paths.append(path)

    # 4) Convertir a array y transponer: shape será (days, num_simulations)
    sims = np.array(sim_paths).T

    # 5) Métricas al final del período
    final_prices = sims[-1, :]
    expected = float(np.mean(final_prices))
    p5       = float(np.percentile(final_prices, 5))
    p95      = float(np.percentile(final_prices, 95))

    # 6) Graficar
    plt.figure(figsize=(10,5))
    plt.plot(sims, lw=0.8, alpha=0.4)
    plt.axhline(expected, linestyle='--', label=f'Esperado: ${expected:.2f}')
    plt.axhline(p5,       linestyle='--', label=f'5%: ${p5:.2f}')
    plt.axhline(p95,      linestyle='--', label=f'95%: ${p95:.2f}')
    plt.title(f"Monte Carlo - {ticker.upper()}")
    plt.xlabel("Días")
    plt.ylabel("Precio simulado")
    plt.legend(); plt.grid(True)

    # 7) Guardar imagen con timestamp para evitar caché
    os.makedirs(STATIC_DIR, exist_ok=True)
    filename = f"montecarlo_{ticker.lower()}_{int(time.time())}.png"
    outpath  = os.path.join(STATIC_DIR, filename)
    plt.savefig(outpath)
    plt.close()

    return filename, expected, p5, p95

import plotly
import plotly.graph_objs as go

def generate_portfolio_plotly(tickers):
    """
    Devuelve (consolidado_json, individual_json) cacheando
    cada descarga 15 min.
    """
    series = []
    for t in tickers:
        time.sleep(0.5)
        data = get_td_timeseries(t, interval='1day', outputsize=100)
        vals = data.get('values', []) or []
        if not vals:
            raise ValueError(f"No hay datos para {t}.")

        df = pd.DataFrame(vals)
        df['close'] = df['close'].astype(float)
        df['date']  = pd.to_datetime(df['datetime'])
        df.sort_values('date', inplace=True)

        df['log_close'] = np.log(df['close'])
        df['Rend']      = df['log_close'].diff().dropna().astype(float)

        # DEBUG opcional:
        print(f"[DEBUG] {t} Rend min/max:", df['Rend'].min(), df['Rend'].max())

        series.append(df.set_index('date')['Rend'].rename(t))

    df_all = pd.concat(series, axis=1).dropna()
    df_all['Portfolio'] = df_all.mean(axis=1)

    # gráfico individual
    traces_ind = [
        go.Scatter(
            x=df_all.index.tolist(),
            y=df_all[t].tolist(),
            mode='lines',
            name=t
        ) for t in tickers
    ]
    fig_ind = go.Figure(
        data=traces_ind,
        layout=go.Layout(
            title='Rend diario por activo',
            xaxis={'title':'Fecha'},
            yaxis={'title':'Rt'},
            template='plotly_white'
        )
    )

    # gráfico consolidado
    trace_cons = go.Scatter(
        x=df_all.index.tolist(),
        y=df_all['Portfolio'].tolist(),
        mode='lines',
        name='Portafolio'
    )
    fig_cons = go.Figure(
        data=[trace_cons],
        layout=go.Layout(
            title='Rend consolidado del portafolio',
            xaxis={'title':'Fecha'},
            yaxis={'title':'Rt'},
            template='plotly_white'
        )
    )

    # serializar ambos
    return (
        json.dumps(fig_cons, cls=plotly.utils.PlotlyJSONEncoder),
        json.dumps(fig_ind, cls=plotly.utils.PlotlyJSONEncoder)
    )

def get_deepseek_interpretation(prompt_text: str) -> str:
    """
    Llama al endpoint de OpenRouter (deepseek/deepseek-r1-zero:free)
    usando chat completions y devuelve la interpretación en texto plano,
    limpiando cualquier envoltura y eliminando símbolos Markdown.
    """
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type":  "application/json"
    }
    payload = {
        "model":    "deepseek/deepseek-r1-zero:free",
        "messages": [
            { "role": "user", "content": prompt_text }
        ]
    }

    resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    # Extrae el contenido
    try:
        content = data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError):
        raise RuntimeError(f"Estructura inesperada en respuesta DeepSeek: {data}")

    # === Limpiar envoltura \boxed{…} si existe ===
    m = re.match(r'^[\\]?boxed\{(.*)\}$', content, flags=re.DOTALL)
    if m:
        content = m.group(1).strip()

    # Elimina símbolos Markdown y caracteres extraños
    content = re.sub(r'[\*\[\]\(\)`]', '', content)
    content = re.sub(r'^[\-\u2022]\s*', '', content, flags=re.MULTILINE)
    content = re.sub(r'>\s*', '', content)
    content = re.sub(r'\s{2,}', ' ', content)

    return content.strip()
    
def plot_portfolio(user_id):
    """
    Genera y guarda dos gráficas:
    1) Consolidada: rendimiento medio del portafolio.
    2) Individual: una línea por cada ticker.
    Reutiliza los mismos datos sin consumir créditos adicionales.
    """
    with SessionLocal() as db:
        items = db.query(PortfolioItem).filter_by(user_id=user_id).all()
    tickers = [i.ticker for i in items]
    if not tickers:
        raise ValueError("Portafolio vacío")

    series = []
    for t in tickers:
        time.sleep(0.5)  # Control de tasa
        data = get_td_timeseries(t, interval='1day', outputsize=100)
        vals = data.get('values') or []
        if not vals:
            raise ValueError(f"No hay datos para {t}.")

        df = pd.DataFrame(vals)
        df['close'] = df['close'].astype(float)
        df['date']  = pd.to_datetime(df['datetime'])
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)
        series.append(np.log(df['close'] / df['close'].shift(1)).rename(t))

    df_all = pd.concat(series, axis=1).dropna()
    df_all['Portfolio'] = df_all.mean(axis=1)

    # Gráfica individual
    plt.figure(figsize=(10,5))
    for t in tickers:
        df_all[t].plot(label=t)
    plt.title('Rendimiento diario por activo')
    plt.xlabel('Fecha'); plt.ylabel('Rt')
    plt.legend()
    os.makedirs(STATIC_DIR, exist_ok=True)
    img_ind = f'portfolio_individual_{int(time.time())}.png'
    plt.savefig(os.path.join(STATIC_DIR, img_ind))
    plt.close()

    # Gráfica consolidada
    plt.figure(figsize=(10,5))
    df_all['Portfolio'].plot(title='Rendimiento diario del Portafolio')
    plt.xlabel('Fecha'); plt.ylabel('Rt')
    img_cons = f'portfolio_consolidado_{int(time.time())}.png'
    plt.savefig(os.path.join(STATIC_DIR, img_cons))
    plt.close()

    return img_cons, img_ind, tickers

def generate_portfolio_pdf(user_id):
    """
    Crea un PDF con la tabla de activos y ambas gráficas:
    - img_cons: rendimiento consolidado
    - img_ind : rendimiento individual
    - tabla de conversión de divisas por activo (con manejo robusto de errores)
    """
    # 1) Obtener usuario
    with SessionLocal() as db:
        user = db.query(User).get(user_id)

    # 2) Desempaquetar correctamente los 3 valores
    img_cons, img_ind, tickers = plot_portfolio(user_id)

    # 3) Preparar documento
    pdf_path = os.path.join(STATIC_DIR, f'portfolio_{user_id}_{int(time.time())}.pdf')
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        rightMargin=50, leftMargin=50,
        topMargin=50, bottomMargin=50
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        'CenteredTitle',
        parent=styles['Heading1'],
        alignment=1, fontSize=18, spaceAfter=12
    ))

    flowables = [
        Paragraph("Portafolio de Inversia", styles['CenteredTitle']),
        Spacer(1, 0.2 * inch),
        Paragraph(f"Nombre: {user.first_name} {user.last_name}", styles['Normal']),
        Paragraph(f"Institución: {user.institucion}", styles['Normal']),
        Spacer(1, 0.3 * inch),
    ]

    # 4) Tabla de activos
    data = [['Activo']] + [[t] for t in tickers]
    table = Table(data, colWidths=[4 * inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2a9d8f')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (1, 0), (-1, -1), colors.whitesmoke),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.gray),
    ]))
    flowables.append(table)
    if flowables and not isinstance(flowables[-1], PageBreak):
        flowables.append(PageBreak())

    # 5) Gráfica consolidada
    flowables.append(
        RLImage(
            os.path.join(STATIC_DIR, img_cons),
            width=6 * inch,
            height=3 * inch
        )
    )
    flowables.append(PageBreak())

    # 6) Gráfica individual
    flowables.append(
        RLImage(
            os.path.join(STATIC_DIR, img_ind),
            width=6 * inch,
            height=3 * inch
        )
    )
    flowables.append(PageBreak())

    # 7) Tabla de conversión de divisas para todos los activos (MEJORADA)
    flowables.append(PageBreak())
    flowables.append(Paragraph("Conversión de divisas para cada activo del portafolio", styles['Heading2']))
    flowables.append(Spacer(1, 0.2 * inch))

    try:
        # Obtener tasas de cambio UNA SOLA VEZ
        print("[INFO] Obteniendo tasas de cambio...")
        rates = get_exchange_rates()
        if not rates or all(v == 0 for v in rates.values()):
            raise ValueError("No se pudieron obtener las tasas de cambio")
        
        headers = ['Ticker', 'USD', 'MXN', 'EUR', 'GBP', 'JPY']
        conv_table = [headers]
        
        successful_tickers = []
        failed_tickers = []

        for i, ticker in enumerate(tickers):
            print(f"[INFO] Procesando ticker {ticker} ({i+1}/{len(tickers)})")
            
            # Añadir delay entre requests para evitar rate limiting
            if i > 0:
                time.sleep(0.5)  # 500ms entre requests
            
            try:
                price = get_ticker_price(ticker)
                if price is None:
                    raise ValueError("No se pudo obtener el precio.")
                
                # Construir fila de la tabla
                fila = [ticker]
                fila.append(f"${price:.2f}")  # USD con formato
                fila.append(f"${price * rates['MXN']:.2f}")
                fila.append(f"€{price * rates['EUR']:.2f}")
                fila.append(f"£{price * rates['GBP']:.2f}")
                fila.append(f"¥{price * rates['JPY']:.0f}")  # JPY sin decimales
                
                conv_table.append(fila)
                successful_tickers.append(ticker)
                print(f"[SUCCESS] {ticker}: ${price:.2f}")
                
            except Exception as e:
                print(f"[ERROR] Fallo con {ticker}: {e}")
                failed_tickers.append(ticker)
                
                # Añadir fila con error en lugar de fallar completamente
                error_row = [ticker, "Error", "Error", "Error", "Error", "Error"]
                conv_table.append(error_row)

        # Crear tabla solo si tenemos al menos un ticker exitoso
        if successful_tickers:
            table2 = Table(conv_table, colWidths=[1.3 * inch] * 6)
            table2.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#264653')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 11),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
                ('BACKGROUND', (0, 1), (-1, -1), colors.whitesmoke),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.gray),
            ]))
            
            flowables.append(table2)
            
            # Información adicional
            today = datetime.now().strftime("%d/%m/%Y")
            nota = Paragraph(
                f"<i>Tasas de cambio obtenidas el {today} desde la API pública de <b>Frankfurter</b>.</i>",
                styles['Normal']
            )
            flowables.append(Spacer(1, 0.15 * inch))
            flowables.append(nota)
            
            # Mostrar información de errores si los hubo
            if failed_tickers:
                error_note = Paragraph(
                    f"<i>Nota: No se pudieron obtener datos para: {', '.join(failed_tickers)}</i>",
                    styles['Normal']
                )
                flowables.append(Spacer(1, 0.1 * inch))
                flowables.append(error_note)
                
            print(f"[INFO] Tabla generada exitosamente. Exitosos: {len(successful_tickers)}, Errores: {len(failed_tickers)}")
        else:
            # Si todos los tickers fallaron
            flowables.append(Paragraph("No se pudieron obtener datos de precios para ningún activo.", styles['Normal']))
            if failed_tickers:
                flowables.append(Paragraph(f"Tickers con error: {', '.join(failed_tickers)}", styles['Normal']))

    except Exception as e:
        print(f"[ERROR CRÍTICO en conversión de divisas] {e}")
        flowables.append(Paragraph("Error crítico al generar la tabla de divisas.", styles['Heading2']))
        flowables.append(Paragraph(f"Detalles del error: {str(e)}", styles['Normal']))

    # 8) Generar PDF
    doc.build(flowables)
    print(f"[INFO] PDF generado exitosamente: {pdf_path}")
    return pdf_path


def send_portfolio_email(to_email, pdf_path):
    """
    Envía el PDF por correo usando SMTP_SSL.
    Lanza RuntimeError si algo falla.
    """
    msg = EmailMessage()
    msg["Subject"] = "Tu Portafolio de Inversia"
    msg["From"]    = EMAIL_USER
    msg["To"]      = to_email
    msg.set_content("Adjunto encontrarás el PDF con tu portafolio.")

    with open(pdf_path, "rb") as f:
        msg.add_attachment(
            f.read(),
            maintype="application",
            subtype="pdf",
            filename=os.path.basename(pdf_path)
        )

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(EMAIL_USER, EMAIL_PASS)
            smtp.send_message(msg)
    except smtplib.SMTPException as e:
        raise RuntimeError(f"Error al enviar correo: {e}")

def send_welcome_email(to_email, first_name):
    """
    Envía un email de bienvenida con HTML/CSS inline,
    sin imágenes, y con banner y botón en color aqua-green (#2a9d8f).
    """
    msg = EmailMessage()
    msg["Subject"] = "¡Bienvenido a Inversia!"
    msg["From"]    = EMAIL_USER
    msg["To"]      = to_email

    # Texto plano (fallback)
    msg.set_content(f"Hola {first_name},\n\n"
                    "¡Gracias por unirte a Inversia! Inicia sesión "
                    "en nuestra plataforma y descubre todas las herramientas "
                    "que tenemos para ti.\n\n"
                    "— El equipo de Inversia")

    login_url = url_for('login', _external=True)

    html = f"""\
    <!DOCTYPE html>
    <html lang="es">
    <head>
      <meta charset="UTF-8">
      <style>
        body {{ margin:0; padding:0; font-family: Arial, sans-serif; background:#f4f4f4; }}
        .container {{ max-width:600px; margin:20px auto; background:#ffffff; border-radius:8px; overflow:hidden; box-shadow:0 2px 8px rgba(0,0,0,0.1); }}
        .banner {{ background:#2a9d8f; color:#ffffff; text-align:center; padding:40px 0; font-size:24px; font-weight:bold; }}
        .content {{ padding:20px; color:#333; }}
        .content h1 {{ margin-top:0; color:#264653; }}
        .btn {{ display:inline-block; margin:20px 0; padding:12px 24px; background:#2a9d8f; color:#fff; text-decoration:none; border-radius:4px; }}
        .footer {{ background:#264653; color:#fff; text-align:center; font-size:12px; padding:10px; }}
      </style>
    </head>
    <body>
      <div class="container">
        <div class="banner">¡Bienvenido a Inversia!</div>
        <div class="content">
          <h1>Hola {first_name},</h1>
          <p>Estamos encantados de darte la bienvenida a <strong>Inversia</strong>, tu nueva plataforma
             para gestionar y analizar tus inversiones de forma fácil y profesional.</p>
          <ul>
            <li>📊 Consulta de cotizaciones en tiempo real</li>
            <li>📈 Gráficas interactivas de rendimiento</li>
            <li>💾 Historial automático de tus búsquedas</li>
            <li>✉️ Compartir tus reportes en PDF por correo</li>
          </ul>
          <a href="{login_url}" class="btn">Iniciar Sesión</a>
          <p>Si tienes dudas o necesitas ayuda, escríbenos a <a href="mailto:{EMAIL_USER}">{EMAIL_USER}</a>.</p>
        </div>
        <div class="footer">
          © 2025 Inversia · Proyecto Académico
        </div>
      </div>
    </body>
    </html>
    """

    msg.add_alternative(html, subtype='html')

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(EMAIL_USER, EMAIL_PASS)
            smtp.send_message(msg)
    except smtplib.SMTPException as e:
        raise RuntimeError(f"Error al enviar correo de bienvenida: {e}")

@app.route('/', methods=['GET','POST'])
def login():
    error = None
    if request.method == 'POST':
        email    = request.form['email'].strip()
        password = request.form['password'].strip()
        if not email or not password:
            error = 'Todos los campos son obligatorios.'
        elif len(email) > MAX_LEN['email'] or len(password) > MAX_LEN['password']:
            error = 'Uno o más campos exceden la longitud máxima permitida.'
        elif re.search(r'\s{2,}', email) or re.search(r'\s{2,}', password):
            error = 'No se permiten espacios consecutivos.'
        if error:
            return render_template('login.html', error=error)
        with SessionLocal() as db:
            u = db.query(User).filter_by(email=email).first()
            if u and check_password_hash(u.password, password):
                session['user_id'] = u.id
                return redirect('/consulta')
        error = 'Credenciales inválidas'
    return render_template('login.html', error=error)

@app.route('/register', methods=['GET', 'POST'])
def register():
    error = None
    if request.method == 'POST':
        # Obtener campos del formulario
        email           = request.form['email'].strip()
        pwd             = request.form['password'].strip()
        confirm_pwd     = request.form['confirm_password'].strip()
        first_name      = request.form['first_name'].strip()
        last_name       = request.form['last_name'].strip()
        motivo          = request.form['reason'].strip()
        institucion     = request.form['institution'].strip()

        # 1) Verificar que contraseña y confirmación coincidan
        if pwd != confirm_pwd:
            error = 'Las contraseñas no coinciden.'
            return render_template('register.html', error=error)

        # 2) Campos obligatorios y sin solo espacios
        raws = {
            'email':       email,
            'password':    pwd,
            'first_name':  first_name,
            'last_name':   last_name,
            'institucion': institucion
        }
        if not all(raws.values()) or not motivo:
            error = 'Todos los campos son obligatorios y no pueden contener solo espacios.'
            return render_template('register.html', error=error)

        # 3) Validar longitud y espacios dobles
        for field, val in raws.items():
            if len(val) > MAX_LEN[field]:
                error = f"El campo {field} no puede exceder {MAX_LEN[field]} caracteres."
                return render_template('register.html', error=error)
            if re.search(r'\s{2,}', val):
                error = f"El campo {field} no puede contener espacios consecutivos."
                return render_template('register.html', error=error)

        # 4) Validar formato de contraseña
        if not re.fullmatch(r'(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[^A-Za-z0-9]).{8,}', pwd):
            error = 'La contraseña no cumple los requisitos (mín 8 char, mayúsc, minúsc, número y símbolo).'
            return render_template('register.html', error=error)

        # 5) Intentar guardar en BD
        with SessionLocal() as db:
            if db.query(User).filter_by(email=email).first():
                error = 'Correo ya registrado.'
                return render_template('register.html', error=error)

            nuevo = User(
                email=email,
                password=generate_password_hash(pwd),
                first_name=first_name,
                last_name=last_name,
                motivo=motivo,
                institucion=institucion
            )
            db.add(nuevo)
            db.commit()
            try:
                send_welcome_email(email, first_name)
            except Exception as e:
                # si falla el envío, solo lo logueamos;
                # no bloqueamos el registro del usuario
                print(f"[WARN] no se pudo enviar email de bienvenida: {e}")
            return redirect(url_for('login'))

    # GET
    return render_template('register.html')

from flask import url_for

def send_reset_email(to_email, code):
    """
    Envía un correo HTML estético para restablecer contraseña,
    con un botón que lleva al usuario a la página de verificación.
    """
    msg = EmailMessage()
    msg["Subject"] = "🔒 Restablece tu contraseña en Inversia"
    msg["From"]    = EMAIL_USER
    msg["To"]      = to_email

    # Fallback de texto plano
    msg.set_content(
        f"Hola,\n\n"
        f"Hemos recibido una solicitud para restablecer la contraseña de tu cuenta ({to_email}).\n"
        f"Tu código de verificación es: {code}\n\n"
        f"Si no solicitaste esto, ignora este mensaje.\n\n"
        f"— El equipo de Inversia"
    )

    # URL a la vista reset_code con el email en query param
    reset_url = url_for('reset_code', email=to_email, _external=True)

    # HTML inline con estilo sobresaliente
    html = f"""\
    <!DOCTYPE html>
    <html lang="es">
    <head>
      <meta charset="UTF-8">
      <style>
        body {{
          margin: 0; padding: 0;
          font-family: 'Arial', sans-serif;
          background: #f0f2f5;
        }}
        .container {{
          max-width: 600px;
          margin: 30px auto;
          background: #ffffff;
          border-radius: 8px;
          overflow: hidden;
          box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }}
        .banner {{
          background: #2a9d8f;
          color: #ffffff;
          text-align: center;
          padding: 24px 0;
          font-size: 22px;
          font-weight: bold;
        }}
        .content {{
          padding: 24px;
          color: #333333;
          line-height: 1.6;
        }}
        .content h2 {{
          margin-top: 0;
          color: #2a9d8f;
        }}
        .code-box {{
          background: #eef7f5;
          border: 2px dashed #2a9d8f;
          padding: 16px;
          text-align: center;
          font-family: 'Courier New', monospace;
          font-size: 20px;
          color: #2a9d8f;
          margin: 20px 0;
          border-radius: 4px;
        }}
        .btn {{
          display: block;
          text-align: center;
          margin: 20px auto;
          padding: 14px 28px;
          background: #2a9d8f;
          color: #ffffff !important;
          text-decoration: none;
          font-size: 16px;
          font-weight: bold;
          border-radius: 4px;
          width: fit-content;
        }}
        .footer {{
          background: #f4f6f8;
          color: #777777;
          text-align: center;
          font-size: 12px;
          padding: 16px;
        }}
      </style>
    </head>
    <body>
      <div class="container">
        <div class="banner">Restablece tu contraseña</div>
        <div class="content">
          <h2>¡Hola!</h2>
          <p>
            Hemos recibido una solicitud para restablecer la contraseña de tu cuenta
            asociada a <strong>{to_email}</strong>.
          </p>
          <p>Tu código de verificación es:</p>
          <div class="code-box">{code}</div>
          <p>
            Haz clic en el siguiente botón para ir al formulario de verificación,
            donde podrás ingresar tu código y elegir una nueva contraseña:
          </p>
          <a href="{reset_url}" class="btn">Verificar y restablecer</a>
          <p>
            Si no solicitaste este cambio, simplemente ignora este correo y tu
            contraseña permanecerá igual.
          </p>
        </div>
        <div class="footer">
          © 2025 Inversia · Proyecto Académico<br>
          ¿Tienes dudas? Contáctanos en <a href="mailto:{EMAIL_USER}">{EMAIL_USER}</a>
        </div>
      </div>
    </body>
    </html>
    """

    msg.add_alternative(html, subtype='html')

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(EMAIL_USER, EMAIL_PASS)
            smtp.send_message(msg)
    except smtplib.SMTPException as e:
        raise RuntimeError(f"Error al enviar correo de restablecimiento: {e}")

@app.route('/forgot_password', methods=['GET','POST'])
def forgot_password():
    error = None
    if request.method == 'POST':
        email = request.form['email'].strip().lower()

        # 1) Validación básica del formato de correo
        if not email or len(email) > MAX_LEN['email'] or re.search(r'\s{2,}', email):
            error = 'Correo inválido.'
            return render_template('forgot_password.html', error=error)

        # 2) Comprobar si el correo existe en la BD
        with SessionLocal() as db:
            user = db.query(User).filter_by(email=email).first()
            if not user:
                error = 'Este correo no está registrado.'
                return render_template('forgot_password.html', error=error)

            # 3) Generar y guardar código único de 6 dígitos
            code = f"{random.randint(0, 999999):06d}"
            db.query(PasswordReset).filter_by(user_id=user.id).delete()
            db.add(PasswordReset(user_id=user.id, code=code))
            db.commit()

        # 4) Enviar el correo con el código
        try:
            send_reset_email(email, code)
        except Exception as e:
            error = str(e)
            return render_template('forgot_password.html', error=error)

        # 5) Redirigir al formulario donde se ingresa el código
        return redirect(url_for('reset_code', email=email))

    # GET
    return render_template('forgot_password.html', error=error)

@app.route('/reset_code', methods=['GET','POST'])
def reset_code():
    error = None
    email = request.form.get('email') or request.args.get('email','')
    if request.method == 'POST':
        code = request.form['code'].strip()
        with SessionLocal() as db:
            user = db.query(User).filter_by(email=email).first()
            pr   = db.query(PasswordReset).filter_by(user_id=user.id, code=code).first() if user else None
            if not pr:
                error = 'Código o correo inválido.'
            else:
                db.delete(pr)
                db.commit()
                session['reset_user_id'] = user.id
                return redirect(url_for('reset_password'))
    return render_template('reset_code.html', error=error, email=email)

@app.route('/reset_password', methods=['GET','POST'])
def reset_password():
    if 'reset_user_id' not in session:
        return redirect(url_for('login'))
    error = None
    if request.method == 'POST':
        pwd     = request.form['password'].strip()
        confirm = request.form['confirm_password'].strip()
        if pwd != confirm:
            error = 'Las contraseñas no coinciden.'
        elif not re.fullmatch(r'(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[^A-Za-z0-9]).{8,}', pwd):
            error = 'La contraseña no cumple los requisitos.'
        else:
            with SessionLocal() as db:
                user = db.query(User).get(session['reset_user_id'])
                user.password = generate_password_hash(pwd)
                db.commit()
            session.pop('reset_user_id', None)
            return redirect(url_for('login'))
    return render_template('reset_password.html', error=error)

@app.route('/historial')
def historial():
    if 'user_id' not in session:
        return redirect('/')
    with SessionLocal() as db:
        historial = (
            db.query(TickerHistory)
              .filter_by(user_id=session['user_id'])
              .order_by(TickerHistory.timestamp.desc())
              .limit(50)
              .all()
        )
    return render_template('historial.html', historial=historial)

# — Chat financiero educativo —
respuestas={
    "¿qué hace apple?": "Apple es una empresa tecnológica que desarrolla hardware como el iPhone, iPad, Mac y servicios como iCloud y Apple Music.",
    "¿qué hace amazon?": "Amazon es una empresa de comercio electrónico y servicios en la nube con presencia global.",
    "¿qué hace tesla?": "Tesla diseña y fabrica automóviles eléctricos y soluciones de energía limpia.",
    "¿qué hace microsoft?": "Microsoft desarrolla software como Windows y Office, además de servicios en la nube como Azure.",
    "¿qué es una acción de crecimiento?": "Son acciones de empresas que reinvierten sus utilidades para expandirse, buscando crecimiento acelerado.",
    "¿qué es una acción de valor?": "Son acciones que cotizan por debajo de su valor intrínseco, ideales para inversionistas a largo plazo.",
    "¿qué es una cartera diversificada?": "Es una combinación de distintos activos financieros para reducir el riesgo total de inversión.",
    "¿qué es el análisis técnico?": "Estudia los movimientos de precios y volumen en los mercados para tomar decisiones de inversión.",
    "¿qué es el análisis fundamental?": "Evalúa la salud financiera de una empresa para estimar su valor real.",
    "¿cuáles son los sectores más estables?": "Salud, bienes de consumo básico y servicios públicos suelen considerarse sectores defensivos.",
    "¿cuáles son los riesgos del mercado?": "Incluyen volatilidad, cambios macroeconómicos, tasas de interés y eventos políticos.",
    "¿qué son las FAANG?": "Es un acrónimo para Facebook, Apple, Amazon, Netflix y Google; empresas tecnológicas líderes.",
    "¿qué es el mercado bursátil?": "Es un sistema donde se compran y venden acciones de empresas que cotizan públicamente.",
    "¿cuándo invertir en acciones?": "Cuando tienes un horizonte de largo plazo y puedes asumir cierta tolerancia al riesgo.",
    "¿qué es un ETF?": "Es un fondo cotizado en bolsa que replica un índice o sector, y se puede comprar como una acción.",
    "¿cómo saber si una empresa es buena para invertir?": "Debes revisar sus estados financieros, su crecimiento, su modelo de negocio y su estabilidad en el mercado.",
    "¿qué es una acción?": "Una acción representa una parte proporcional del capital social de una empresa.",
    "¿qué es una inversión?": "Una inversión es colocar dinero en un instrumento esperando obtener un rendimiento.",
    "¿qué es un dividendo?": "Es la parte de las ganancias que una empresa reparte entre sus accionistas.",
    "¿qué es el riesgo financiero?": "Es la probabilidad de perder parte o la totalidad del dinero invertido.",
    "¿qué es un bono?": "Un bono es un instrumento de deuda emitido por una empresa o gobierno.",
    "¿qué es una tasa de interés?": "Es el porcentaje que se paga por usar dinero ajeno durante un periodo de tiempo.",
    "¿qué es la inflación?": "Es el aumento generalizado de los precios con el tiempo, lo que reduce el poder adquisitivo del dinero.",
    "¿qué es un fondo de inversión?": "Es un vehículo financiero que agrupa dinero de varios inversionistas para invertirlo de forma diversificada.",
    "¿qué es el perfil de riesgo?": "Es una evaluación de la tolerancia del inversionista frente a posibles pérdidas.",
    "¿qué es el plazo fijo?": "Es una inversión bancaria donde dejas tu dinero un tiempo determinado a cambio de intereses.",
    "¿qué es un activo financiero?": "Es cualquier recurso con valor económico que puede generar ingresos o beneficios en el futuro.",
    "¿qué es el apalancamiento financiero?": "Es usar dinero prestado para aumentar el potencial de rentabilidad de una inversión.",
    "¿qué es el patrimonio neto?": "Es la diferencia entre tus activos (lo que posees) y tus pasivos (lo que debes).",
    "¿qué es una IPO?": "Es una Oferta Pública Inicial, cuando una empresa vende acciones al público por primera vez.",
    "¿qué es el valor intrínseco de una acción?": "Es el valor real estimado de una acción basado en sus fundamentos, no en su precio de mercado.",
    "¿qué es el análisis técnico?": "Es el estudio de gráficos e indicadores para predecir el comportamiento de precios.",
    "¿qué es el análisis fundamental?": "Es el análisis de la salud financiera de una empresa para estimar su valor real.",
    "¿qué es la diversificación?": "Es distribuir tu dinero en diferentes activos para reducir el riesgo global.",
    "¿cuál es la diferencia entre renta fija y renta variable?": "Renta fija ofrece pagos conocidos (como bonos), renta variable depende del desempeño (como acciones).",
    "¿qué es el mercado alcista?": "Es un periodo sostenido de crecimiento en los precios de los activos.",
    "¿qué es el mercado bajista?": "Es un periodo sostenido de caída en los precios de los activos.",
    "¿qué es un split de acciones?": "Es cuando una empresa divide sus acciones en más unidades sin cambiar su valor total.",
    "¿qué es la volatilidad?": "Es la medida de cuánto varía el precio de un activo en un periodo de tiempo.",
}

tips = [
    "Piensa en tus objetivos a largo plazo antes de invertir.",
    "Diversificar reduce el riesgo.",
    "Evita decisiones impulsivas cuando el mercado se mueve.",
    "Ahorra al menos el 10% de tus ingresos.",
    "Invierte solo lo que estés dispuesta a perder.",
    "Consulta fuentes confiables antes de tomar decisiones financieras.",
]

def buscar_respuesta(pregunta):
    for clave in respuestas:
        if clave in pregunta:
            return respuestas[clave]
    return "Lo siento, no tengo una respuesta específica para esa pregunta."

@app.route('/chatfinanzas', methods=['GET', 'POST'])
def chatfinanzas():
    respuesta = ""
    tip = random.choice(tips)

    if request.method == 'POST':
        # Tomar pregunta del texto o del dropdown
        pregunta = request.form.get('pregunta_texto', '').strip()
        if not pregunta:
            pregunta = request.form.get('pregunta_dropdown', '').strip()

        if pregunta:
            session['ultima_consulta'] = pregunta
            respuesta = buscar_respuesta(pregunta.lower())
        else:
            respuesta = "Por favor selecciona o escribe una pregunta para continuar."

    return render_template(
        'chatfinanzas.html',
        respuesta=respuesta,
        tip=tip,
        respuestas=respuestas
    )

TICKERS_CACHE_PATH = "valid_tickers.json"
TICKERS_CACHE_TTL = timedelta(days=1)

def get_valid_tickers():
    """
    Devuelve un conjunto de tickers válidos.
    Usa cache en disco (24h) para evitar llamar a Twelve Data innecesariamente.
    """
    # 1. Revisar si el archivo cache existe y es reciente
    if os.path.exists(TICKERS_CACHE_PATH):
        last_modified = datetime.fromtimestamp(os.path.getmtime(TICKERS_CACHE_PATH))
        if datetime.now() - last_modified < TICKERS_CACHE_TTL:
            try:
                with open(TICKERS_CACHE_PATH, 'r') as f:
                    tickers = json.load(f)
                return set(tickers)
            except Exception as e:
                print(f"[CACHE] Error leyendo archivo de tickers: {e}")

    # 2. Si no existe o está desactualizado, descargar desde Twelve Data
    try:
        print("[INFO] Descargando lista de tickers desde Twelve Data...")
        resp = requests.get(TD_STOCKS_URL, params={'apikey': TD_API_KEY}, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        stocks = data.get('data', [])
        if not isinstance(stocks, list):
            raise ValueError("Formato inesperado en respuesta de Twelve Data")

        symbols = sorted(set(item['symbol'] for item in stocks))
        with open(TICKERS_CACHE_PATH, 'w') as f:
            json.dump(symbols, f)

        return set(symbols)

    except Exception as e:
        print(f"[ERROR] No se pudo actualizar lista de tickers: {e}")
        # Como fallback, intenta devolver lo que ya hay en cache aunque esté viejo
        if os.path.exists(TICKERS_CACHE_PATH):
            with open(TICKERS_CACHE_PATH, 'r') as f:
                return set(json.load(f))
        return set()

@app.route('/consulta', methods=['GET', 'POST'])
def consulta():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    error          = None
    ticker         = ''
    conversiones   = {}
    grafico_json   = None
    interpretation = None

    if request.method == 'POST':
        action = request.form.get('action')   # "consult" o "deepseek"
        raw    = request.form.get('ticker', '')
        ticker = raw.strip().upper()

        # Validaciones básicas
        if not ticker:
            error = 'El ticker es obligatorio.'
        elif len(ticker) > MAX_LEN['ticker']:
            error = 'El ticker es demasiado largo.'
        elif re.search(r'\s{2,}', raw):
            error = 'No se permiten espacios consecutivos.'

        if not error:
            try:
                validos = get_valid_tickers()
                if ticker not in validos:
                    error = f"Ticker '{ticker}' no está en la lista oficial."
                else:
                    # Gráfico interactivo
                    grafico_json = fetch_and_plot_td_plotly(ticker)

                    # Precio y conversiones
                    data = get_td_timeseries(ticker, interval='1day', outputsize=1)
                    price = float(data['values'][0]['close'])
                    rates = get_exchange_rates()
                    conversiones = {
                        'USD': round(price, 2),
                        'MXN': round(price * rates['MXN'], 2),
                        'EUR': round(price * rates['EUR'], 2),
                        'GBP': round(price * rates['GBP'], 2),
                        'JPY': round(price * rates['JPY'], 2),
                    }

                    # Guardar historial
                    if action == 'consult':
                        with SessionLocal() as db:
                            db.add(TickerHistory(user_id=session['user_id'], ticker=ticker))
                            db.commit()

                    # Análisis experto con DeepSeek
                    if action == 'deepseek':
                        try:
                            # 1) Descarga datos completos para estadísticas
                            vals = get_td_timeseries(ticker, interval='1day', outputsize=100)['values']
                            df_vals = pd.DataFrame(vals)
                            df_vals['close'] = df_vals['close'].astype(float)
                            df_vals['date']  = pd.to_datetime(df_vals['datetime'])
                            df_vals.sort_values('date', inplace=True)
                            df_vals['Rendimiento'] = np.log(df_vals['close'] / df_vals['close'].shift(1))
                            df_vals.dropna(inplace=True)

                            # 2) Calcula estadísticas clave
                            avg_ret = round(df_vals['Rendimiento'].mean(), 4)
                            vol     = round(df_vals['Rendimiento'].std(), 4)
                            min_ret = round(df_vals['Rendimiento'].min(), 4)
                            max_ret = round(df_vals['Rendimiento'].max(), 4)

                            # 3) Construye prompt “plano”
                            prompt = (
                                f"Como experto financiero y actuario, analiza en texto plano "
                                f"los siguientes datos sobre {ticker}: retorno promedio diario {avg_ret}, "
                                f"volatilidad {vol}, mínimo {min_ret}, máximo {max_ret}, "
                                f"y la tabla de conversiones USD {conversiones['USD']}, "
                                f"MXN {conversiones['MXN']}, EUR {conversiones['EUR']}, "
                                f"GBP {conversiones['GBP']}, JPY {conversiones['JPY']}. "
                                "Usa solo texto corrido, sin asteriscos, sin guiones ni corchetes, "
                                "sin viñetas, sin símbolos especiales."
                            )

                            interpretation = get_deepseek_interpretation(prompt)
                        except Exception as e_int:
                            interpretation = f"No se pudo obtener interpretación: {e_int}"

            except Exception as e:
                error = f"Error validando ticker: {e}"

    return render_template(
        'consulta.html',
        error=error,
        grafico_json=grafico_json,
        ticker=ticker,
        conversiones=conversiones,
        interpretation=interpretation
    )

@app.route('/add', methods=['POST'])
def add_portfolio():
    if 'user_id' not in session:
        return redirect('/')
    raw    = request.form.get('ticker', '')
    ticker = raw.strip().upper()
    error  = None

    # 1) validaciones básicas de formato
    if not ticker:
        error = 'El ticker es obligatorio.'
    elif len(ticker) > MAX_LEN['ticker']:
        error = 'El ticker es demasiado largo.'
    elif re.search(r'\s{2,}', raw):
        error = 'No se permiten espacios consecutivos.'
    else:
        # 2) validación contra la lista oficial
        try:
            validos = get_valid_tickers()
        except Exception as e:
            error = f"Error validando lista de tickers: {e}"
        else:
            if ticker not in validos:
                error = f"Ticker '{ticker}' no está en la lista oficial."
            else:
                # 3) si pasa validación, guardamos en BD
                with SessionLocal() as db:
                    item = PortfolioItem(user_id=session['user_id'], ticker=ticker)
                    db.add(item)
                    try:
                        db.commit()
                    except IntegrityError:
                        db.rollback()  # ya existía, lo ignoramos

    if error:
        # 4) en caso de error, re-renderizamos portafolio con lista + gráficas (si hay)
        with SessionLocal() as db:
            items = db.query(PortfolioItem).filter_by(user_id=session['user_id']).all()
        tickers = [i.ticker for i in items]

        try:
            cons, ind, _ = plot_portfolio(session['user_id'])
        except:
            cons, ind = None, None

        return render_template(
            'portfolio.html',
            error=error,
            tickers=tickers,
            consolidated=cons,
            individual=ind
        )

    return redirect('/portfolio')

@app.route('/portfolio', methods=['GET', 'POST'])
def portfolio():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    msg = None
    error = None
    generate_analysis = False
    interp_cons = None
    interp_ind = None

    # 1) envío de PDF
    if request.method == 'POST' and 'email' in request.form:
        raw_email = request.form['email'].strip()
        try:
            pdf_path = generate_portfolio_pdf(session['user_id'])
            send_portfolio_email(raw_email, pdf_path)
            msg = f'Enviado a {raw_email} correctamente.'
        except Exception as e:
            msg = str(e)

    # 2) botón de análisis experto
    if request.method == 'POST' and request.form.get('action') == 'deepseek':
        generate_analysis = True

    # 3) obtener tickers
    with SessionLocal() as db:
        items = db.query(PortfolioItem).filter_by(user_id=session['user_id']).all()
    tickers = [i.ticker for i in items]
    if not tickers:
        error = "Portafolio vacío"
        return render_template('portfolio.html', error=error, tickers=[])

    # 4) validación de tickers
    try:
        validos = get_valid_tickers()
        invalidos = [t for t in tickers if t not in validos]
        if invalidos:
            error = (
                f"Ticker '{invalidos[0]}' no está en la lista oficial."
                if len(invalidos) == 1 else
                f"Tickers inválidos: {', '.join(invalidos)}."
            )
            return render_template('portfolio.html', error=error, tickers=tickers)
    except Exception as e:
        return render_template('portfolio.html', error=f"Error validando tickers: {e}", tickers=tickers)

    # 5) generar gráficas Plotly
    try:
        consolidated_json, individual_json = generate_portfolio_plotly(tickers)
    except Exception as e:
        return render_template('portfolio.html', error=f"Error generando gráficas: {e}", tickers=tickers)

    # 6) interpretaciones con DeepSeek, ahora con estadísticas concretas
    if generate_analysis:
        # reconstruimos los retornos para poder calcular stats
        series = []
        for t in tickers:
            data = get_td_timeseries(t, interval='1day', outputsize=100)['values']
            df = pd.DataFrame(data)
            df['close'] = df['close'].astype(float)
            df['date']  = pd.to_datetime(df['datetime'])
            df.sort_values('date', inplace=True)
            df['Rend'] = np.log(df['close'] / df['close'].shift(1))
            series.append(df.set_index('date')['Rend'].rename(t))
        df_all = pd.concat(series, axis=1).dropna()
        df_all['Portfolio'] = df_all.mean(axis=1)

        # Análisis consolidado
        try:
            avg_p = round(df_all['Portfolio'].mean(), 4)
            vol_p = round(df_all['Portfolio'].std(), 4)
            min_p = round(df_all['Portfolio'].min(), 4)
            max_p = round(df_all['Portfolio'].max(), 4)

            prompt1 = (
                f"Como experto financiero y actuario, analiza en texto plano el gráfico "
                f"consolidado de rendimiento de mi portafolio con retorno promedio {avg_p}, "
                f"volatilidad {vol_p}, mínimo {min_p} y máximo {max_p}. "
                "Usa solo texto corrido sin símbolos especiales."
            )
            interp_cons = get_deepseek_interpretation(prompt1)
        except Exception as e_int:
            interp_cons = f"No se pudo obtener interpretación consolidada: {e_int}"

        # Análisis individual
        try:
            stats = []
            for t in tickers:
                m = round(df_all[t].mean(), 4)
                v = round(df_all[t].std(), 4)
                stats.append(f"{t} retorno promedio {m} y volatilidad {v}")
            stats_str = ", ".join(stats)

            prompt2 = (
                f"Como experto financiero y actuario, analiza en texto plano el gráfico "
                f"individual de rendimiento por activo. Datos por ticker: {stats_str}. "
                "Usa solo texto corrido sin símbolos especiales."
            )
            interp_ind = get_deepseek_interpretation(prompt2)
        except Exception as e_int:
            interp_ind = f"No se pudo obtener interpretación individual: {e_int}"

    return render_template(
        'portfolio.html',
        tickers=tickers,
        message=msg,
        error=None,
        consolidated_json=consolidated_json,
        individual_json=individual_json,
        interp_cons=interp_cons,
        interp_ind=interp_ind
    )

@app.route('/montecarlo', methods=['GET'])
def montecarlo_tutorial():
    """
    Muestra la pantalla de introducción a Monte Carlo.
    """
    return render_template('montecarlo_tutorial.html')

@app.route('/montecarlo/run', methods=['GET','POST'])
def montecarlo_run():
    """
    Ejecuta la simulación Monte Carlo usando yfinance.
    """
    error   = None
    imagen  = None
    ticker  = ''
    esperado = p5 = p95 = None

    if request.method == 'POST':
        raw    = request.form.get('ticker', '')
        days   = int(request.form.get('days', 30))
        sims   = int(request.form.get('num_sim', 100))
        ticker = raw.strip().upper()

        if not ticker:
            error = "El ticker es obligatorio."
        else:
            try:
                # => tu función revisada de MC <
                imagen, esperado, p5, p95 = monte_carlo_simulation_yf(
                    ticker, days=days, num_simulations=sims
                )
                if not imagen:
                    error = f"No se pudo generar la simulación para '{ticker}'."
            except Exception as e:
                error = str(e)

    return render_template(
        'monteCarlo.html',
        ticker=ticker,
        imagen=imagen,
        error=error,
        esperado=esperado,
        p5=p5,
        p95=p95
    )

@app.route('/delete', methods=['POST'])
def delete_portfolio():
    if 'user_id' not in session:
        return redirect('/')
    # Tomamos el ticker a eliminar
    raw = request.form.get('ticker', '')
    ticker = raw.strip().upper()
    if ticker:
        with SessionLocal() as db:
            item = (
                db.query(PortfolioItem)
                  .filter_by(user_id=session['user_id'], ticker=ticker)
                  .first()
            )
            if item:
                db.delete(item)
                db.commit()
    return redirect('/portfolio')

@app.route('/backtest', methods=['GET'])
def backtest_tutorial():
    """
    Muestra la página de introducción/tutoria
    antes de ejecutar el backtest real.
    """
    return render_template('backtest_tutorial.html')

import pandas as pd
import numpy as np
import plotly.graph_objs as go
import json
import yfinance as yf
from flask import request, render_template

@app.route('/backtest/run', methods=['GET','POST'])
def backtest_run():
    """
    Ejecuta realmente el backtest SMA crossover usando yfinance.
    """
    error      = None
    graph_json = None
    ticker     = ''
    fast       = 20
    slow       = 50
    capital    = 10000.0
    cagr = max_dd = sharpe = None

    if request.method == 'POST':
        # 1) Leer inputs
        ticker  = request.form['ticker'].strip().upper()
        fast    = int(request.form['fast'])
        slow    = int(request.form['slow'])
        capital = float(request.form['capital'])

        # 2) Validaciones
        if not ticker:
            error = "Ticker obligatorio."
        elif fast >= slow:
            error = "La SMA rápida debe ser menor que la lenta."

        if not error:
            try:
                # 3) Descargar con yfinance
                df = yf.download(
                    tickers=ticker,
                    period="1y",
                    interval="1d",
                    progress=False,
                    auto_adjust=False,
                    threads=False
                )
                if df.empty or 'Close' not in df:
                    raise ValueError("No hay datos con yfinance.")

                # 4) Preparar DataFrame
                df = df[['Close']].rename(columns={'Close':'close'})
                df.index.name = 'date'
                df.sort_index(inplace=True)

                # 5) Calcular SMAs
                df[f'SMA{fast}'] = df['close'].rolling(fast).mean()
                df[f'SMA{slow}'] = df['close'].rolling(slow).mean()

                # 6) Señal (shift sobre Serie, no ndarray)
                signals = df[f'SMA{fast}'] > df[f'SMA{slow}']
                df['signal'] = signals.shift(1).fillna(False).astype(float)

                # 7) Retornos diarios
                df['ret'] = df['close'].pct_change().fillna(0.0)

                # 8) Equity curves
                df['strat_ret'] = df['ret'] * df['signal']
                df['equity']    = (1 + df['strat_ret']).cumprod() * capital
                df['buy_hold']  = (1 + df['ret']).cumprod() * capital

                # 9) Métricas
                total_days   = df.shape[0]
                years        = total_days / 252
                final_equity = df['equity'].iloc[-1]
                cagr         = (final_equity / capital)**(1/years) - 1

                roll_max = df['equity'].cummax()
                drawdown = (df['equity'] - roll_max) / roll_max
                max_dd   = drawdown.min()

                sharpe = (
                    df['strat_ret'].mean() /
                    df['strat_ret'].std(ddof=1)
                ) * np.sqrt(252)

                # 10) Gráfico Plotly
                trace_strat = go.Scatter(
                    x=df.index.tolist(),
                    y=df['equity'].tolist(),
                    mode='lines', name='Estrategia'
                )
                trace_bh = go.Scatter(
                    x=df.index.tolist(),
                    y=df['buy_hold'].tolist(),
                    mode='lines', name='Buy & Hold'
                )
                layout = go.Layout(
                    title=f"Backtest {ticker} SMA{fast}/{slow}",
                    xaxis={'title':'Fecha'},
                    yaxis={'title':'Valor (USD)'},
                    template='plotly_white'
                )
                fig = go.Figure(data=[trace_strat, trace_bh], layout=layout)
                graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

            except Exception as e:
                error = f"No se pudo backtestear '{ticker}': {e}"

    return render_template(
        'backtest.html',
        error=error,
        graph_json=graph_json,
        ticker=ticker,
        fast=fast,
        slow=slow,
        capital=capital,
        cagr=cagr or 0,
        max_dd=max_dd or 0,
        sharpe=sharpe or 0
    )

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/')

if __name__ == '__main__':
    app.run(debug=True)