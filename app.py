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


@lru_cache(maxsize=100)
def get_ticker_price(ticker):
    try:
        params = {
            'symbol': ticker,
            'interval': '1day',
            'outputsize': 1,
            'apikey': TD_API_KEY,
            'format': 'JSON'
        }
        r = requests.get(TD_URL, params=params).json()
        return float(r['values'][0]['close'])
    except Exception as e:
        print(f"[ERROR] No se pudo obtener el precio para {ticker}: {e}")
        return 0.0


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

import plotly
import plotly.graph_objs as go
import json

import requests
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import json

def fetch_and_plot_td_plotly(ticker: str) -> str:
    """
    Descarga datos de TwelveData, calcula log-retornos y devuelve
    un JSON listo para Plotly, con y-values como floats.
    """
    # 1) Llamada a la API
    params = {
        'symbol':     ticker,
        'interval':   '1day',
        'outputsize': 100,
        'apikey':     TD_API_KEY,
        'format':     'JSON'
    }
    resp = requests.get(TD_URL, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    values = data.get('values', [])
    if not values:
        raise ValueError(f"No hay datos para {ticker}")

    # 2) Armar DataFrame
    df = pd.DataFrame(values)
    df['close'] = df['close'].astype(float)
    df['date']  = pd.to_datetime(df['datetime'])
    df.sort_values('date', inplace=True)

    # 3) Calcular log-retornos
    df['log_close']   = np.log(df['close'])
    df['Rendimiento'] = df['log_close'].diff()
    df = df.dropna(subset=['Rendimiento'])

    # 4) Forzar tipo float y debug rápido
    df['Rendimiento'] = df['Rendimiento'].astype(float)
    print(f"[DEBUG] {ticker} Rend min/max:", 
          df['Rendimiento'].min(), df['Rendimiento'].max())

    # 5) Construir la figura Plotly
    trace = go.Scatter(
        x=df['date'].tolist(),
        y=df['Rendimiento'].tolist(),   # <-- lista de floats
        mode='lines',
        name=f'Rendimiento diario de {ticker}'
    )
    layout = go.Layout(
        title=f'Rendimiento diario de {ticker}',
        xaxis=dict(title='Fecha'),
        yaxis=dict(title='Rt'),
        template='plotly_white'
    )
    fig = go.Figure(data=[trace], layout=layout)

    # 6) Serializar a JSON
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def monte_carlo_simulation_yf(ticker, days=30, num_simulations=100):
    import yfinance as yf
    import os

    # Si no puedes importar STATIC_DIR directamente:
    STATIC_DIR = os.path.join(os.path.dirname(__file__), 'static')

    try:
        print(f"[INFO] Simulación Monte Carlo: {ticker}, días={days}, simulaciones={num_simulations}")
        data = yf.download(ticker, period='60d', interval='1d', progress=False)
        if data.empty:
            raise ValueError("No se obtuvieron datos históricos")

        close_prices = data['Close'].dropna().to_numpy()
        if len(close_prices) < 2:
            raise ValueError("Datos insuficientes para simular.")

        last_price = close_prices[-1]
        returns = np.diff(np.log(close_prices))
        mu = returns.mean()
        sigma = returns.std()

        print(f"[INFO] Último precio={last_price:.2f}, mu={mu:.5f}, sigma={sigma:.5f}")

        simulations = np.zeros((days, num_simulations))
        for sim in range(num_simulations):
            prices = [last_price]
            for _ in range(1, days):
                shock = np.random.normal(mu - 0.5 * sigma**2, sigma)
                prices.append(prices[-1] * np.exp(shock))
            simulations[:, sim] = prices

        final_prices = simulations[-1, :]
        expected = np.mean(final_prices)
        p5 = np.percentile(final_prices, 5)
        p95 = np.percentile(final_prices, 95)

        plt.figure(figsize=(10, 5))
        plt.plot(simulations, lw=0.8, alpha=0.4)
        plt.axhline(expected, color='blue', linestyle='--', label=f'Esperado: ${expected:.2f}')
        plt.axhline(p5, color='red', linestyle='--', label=f'5%: ${p5:.2f}')
        plt.axhline(p95, color='green', linestyle='--', label=f'95%: ${p95:.2f}')
        plt.title(f"Simulación Monte Carlo - {ticker.upper()}")
        plt.xlabel("Días")
        plt.ylabel("Precio simulado")
        plt.legend()
        plt.grid(True)

        os.makedirs(STATIC_DIR, exist_ok=True)
        filename = f"montecarlo_{ticker}.png"
        output_path = os.path.join(STATIC_DIR, filename)
        plt.savefig(output_path)
        plt.close()

        print(f"[INFO] Imagen guardada: {output_path}")
        return filename, expected, p5, p95

    except Exception as e:
        print(f"[ERROR Monte Carlo] {e}")
        return None, None, None, None

import requests
import pandas as pd
import numpy as np
import plotly
import plotly.graph_objs as go
import json

def generate_portfolio_plotly(tickers):
    """
    Devuelve dos cadenas JSON para Plotly:
      1) gráfico consolidado: rendimiento promedio del portafolio
      2) gráfico individual: línea de rendimiento por activo
    Ambas usan log-retornos (no precios) y pasan listas de floats a Plotly.
    """
    series = []
    for t in tickers:
        # 1) Descargar datos
        params = {
            'symbol':     t,
            'interval':   '1day',
            'outputsize': 100,
            'apikey':     TD_API_KEY,
            'format':     'JSON'
        }
        resp = requests.get(TD_URL, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        vals = data.get('values', []) or []
        if not vals:
            raise ValueError(f"No hay datos para {t}.")

        # 2) Crear DataFrame y preparar fechas/precios
        df = pd.DataFrame(vals)
        df['close'] = df['close'].astype(float)
        df['date']  = pd.to_datetime(df['datetime'])
        df.sort_values('date', inplace=True)

        # 3) Calcular log-retornos
        df['log_close'] = np.log(df['close'])
        df['Rend']      = df['log_close'].diff()
        df.dropna(subset=['Rend'], inplace=True)

        # 4) Debug (opcional): checar rango de retornos
        print(f"[DEBUG] {t} Rend min/max:", df['Rend'].min(), df['Rend'].max())

        # 5) Añadir la serie de retornos al array
        series.append(df.set_index('date')['Rend'].astype(float).rename(t))

    # 6) Concatenar todas las series y calcular la media (portafolio)
    df_all = pd.concat(series, axis=1)
    df_all.dropna(inplace=True)
    df_all['Portfolio'] = df_all.mean(axis=1)

    # 7) Construir gráfico individual
    traces_ind = []
    for t in tickers:
        traces_ind.append(go.Scatter(
            x=df_all.index.tolist(),
            y=df_all[t].tolist(),       # lista de floats, no Series ni strings
            mode='lines',
            name=t
        ))
    fig_ind = go.Figure(
        data=traces_ind,
        layout=go.Layout(
            title='Rendimiento diario por activo',
            xaxis=dict(title='Fecha'),
            yaxis=dict(title='Rt'),
            template='plotly_white'
        )
    )

    # 8) Construir gráfico consolidado
    trace_cons = go.Scatter(
        x=df_all.index.tolist(),
        y=df_all['Portfolio'].tolist(),
        mode='lines',
        name='Portafolio'
    )
    fig_cons = go.Figure(
        data=[trace_cons],
        layout=go.Layout(
            title='Rendimiento consolidado del portafolio',
            xaxis=dict(title='Fecha'),
            yaxis=dict(title='Rt'),
            template='plotly_white'
        )
    )

    # 9) Serializar a JSON para inyectar en la plantilla
    return (
        json.dumps(fig_cons, cls=plotly.utils.PlotlyJSONEncoder),
        json.dumps(fig_ind, cls=plotly.utils.PlotlyJSONEncoder)
    )

def get_deepseek_interpretation(prompt_text: str) -> str:
    """
    Llama al endpoint de OpenRouter (deepseek/deepseek-r1-zero:free)
    usando chat completions y devuelve la interpretación, limpiando
    cualquier envoltura \boxed{...}.
    """
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type":  "application/json",
        "HTTP-Referer":  "https://tu-sitio.com",  # opcional
        "X-Title":       "Inversia"               # opcional
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

    try:
        content = data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError):
        raise RuntimeError(f"Estructura inesperada en respuesta DeepSeek: {data}")

    # === Limpiar envoltura \boxed{...} si existe ===
    # Coincide \boxed{cualquier texto} y extrae solo 'cualquier texto'
    m = re.match(r'^[\\]?boxed\{(.*)\}$', content, flags=re.DOTALL)
    if m:
        content = m.group(1).strip()

    return content
    
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
        params = {
            'symbol':     t,
            'interval':   '1day',
            'outputsize': 100,
            'apikey':     TD_API_KEY,
            'format':     'JSON'
        }
        resp = requests.get(TD_URL, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get('status') == 'error':
            raise ValueError(f"API error para {t}: {data.get('message','desconocido')}")
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

import datetime
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
            today = datetime.datetime.now().strftime("%d/%m/%Y")
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

@lru_cache(maxsize=1)
def get_valid_tickers():
    """
    Descarga y cachea la lista de tickers válidos de Twelve Data.
    """
    resp = requests.get(TD_STOCKS_URL, params={'apikey': TD_API_KEY}, timeout=10)
    resp.raise_for_status()
    data = resp.json()               # { "data": [ { "symbol": "...", ... }, ... ], ... }
    stocks = data.get('data', [])
    if not isinstance(stocks, list):
        raise ValueError("Formato inesperado al obtener lista de tickers")
    return { item['symbol'] for item in stocks }

@app.route('/consulta', methods=['GET', 'POST'])
def consulta():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    error          = None
    resultado      = None
    ticker         = ''
    conversiones   = {}
    interpretation = None
    grafico_json   = None  # ← importante para evitar el UnboundLocalError

    if request.method == 'POST':
        action = request.form.get('action')  # "consult" o "deepseek"
        raw    = request.form.get('ticker', '')
        ticker = raw.strip().upper()

        # validaciones básicas
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
                    # ✅ Gráfico interactivo con Plotly
                    grafico_json = fetch_and_plot_td_plotly(ticker)

                    # Precio y conversiones
                    resp_json = requests.get(
                        TD_URL,
                        params={
                            'symbol': ticker,
                            'interval': '1day',
                            'outputsize': 1,
                            'apikey': TD_API_KEY,
                            'format': 'JSON'
                        },
                        timeout=10
                    ).json()
                    price = float(resp_json['values'][0]['close'])
                    rates = get_exchange_rates()
                    conversiones = {
                        'USD': round(price, 2),
                        'MXN': round(price * rates['MXN'], 2),
                        'EUR': round(price * rates['EUR'], 2),
                        'GBP': round(price * rates['GBP'], 2),
                        'JPY': round(price * rates['JPY'], 2),
                    }

                    # Guardar historial si es consulta normal
                    if action == 'consult':
                        with SessionLocal() as db:
                            db.add(TickerHistory(user_id=session['user_id'], ticker=ticker))
                            db.commit()

                    # Interpretación solo si se pidió análisis experto
                    if action == 'deepseek':
                        try:
                            prompt = (
                                f"Como experto financiero y actuario, analiza en texto plano "
                                f"el gráfico de rendimiento diario de {ticker} y la tabla de conversiones {conversiones}. "
                                "Sin código, sin fences, sin bullets Markdown."
                            )
                            interpretation = get_deepseek_interpretation(prompt)
                        except Exception as e_int:
                            interpretation = f"No se pudo obtener interpretación: {e_int}"

            except Exception as e:
                error = f"Error validando ticker: {e}"

    return render_template(
        'consulta.html',
        error=error,
        resultado=None,
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
    generate_analysis = False

    # 1) manejo de envío de PDF
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

    # 4) validar tickers
    try:
        validos = get_valid_tickers()
        invalidos = [t for t in tickers if t not in validos]
        if invalidos:
            error = (
                f"Ticker '{invalidos[0]}' no está en la lista oficial."
                if len(invalidos) == 1 else
                f"Los siguientes tickers no están en la lista oficial: {', '.join(invalidos)}."
            )
            return render_template('portfolio.html', error=error, tickers=tickers)
    except Exception as e:
        return render_template('portfolio.html', error=f"Error validando tickers: {e}", tickers=tickers)

    # 5) generar gráficas Plotly
    try:
        consolidated_json, individual_json = generate_portfolio_plotly(tickers)
    except Exception as e:
        return render_template('portfolio.html', error=f"Error generando gráficas: {e}", tickers=tickers)

    # 6) interpretaciones
    interp_cons = interp_ind = None
    if generate_analysis:
        try:
            prompt1 = (
                "Como experto financiero y actuario, ofrece un análisis DEL GRÁFICO CONSOLIDADO "
                "DE RENDIMIENTO en puro texto plano, sin código ni sintaxis."
            )
            interp_cons = get_deepseek_interpretation(prompt1)
        except Exception as e_int:
            interp_cons = f"No se pudo obtener interpretación: {e_int}"

        try:
            prompt2 = (
                "Como experto financiero y actuario, ofrece un análisis DEL GRÁFICO INDIVIDUAL "
                "DE RENDIMIENTO POR ACTIVO en puro texto plano, sin código ni sintaxis."
            )
            interp_ind = get_deepseek_interpretation(prompt2)
        except Exception as e_int:
            interp_ind = f"No se pudo obtener interpretación: {e_int}"

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

@app.route('/montecarlo', methods=['GET', 'POST'])
def montecarlo():
    if 'user_id' not in session:
        return redirect('/')

    error = None
    imagen = None
    ticker = ''
    esperado = p5 = p95 = None

    if request.method == 'POST':
        raw = request.form.get('ticker', '')
        ticker = raw.strip().upper()

        if not ticker:
            error = "El ticker es obligatorio."
        else:
            imagen, esperado, p5, p95 = monte_carlo_simulation_yf(ticker)
            if imagen is None:
                error = f"No se pudo generar la simulación para '{ticker}'."

    return render_template('montecarlo.html',
                           ticker=ticker,
                           imagen=imagen,
                           error=error,
                           esperado=esperado,
                           p5=p5,
                           p95=p95)

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

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/')

if __name__ == '__main__':
    app.run(debug=True)
