# app.py
import os
import time
import re
import requests
from requests import RequestException
from functools import lru_cache
import numpy as np
import pandas as pd
from werkzeug.security import generate_password_hash, check_password_hash

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


def generate_portfolio_pdf(user_id):
    """
    Crea un PDF con la tabla de activos y ambas gráficas:
    - img_cons: rendimiento consolidado
    - img_ind : rendimiento individual
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
        Spacer(1, 0.2*inch),
        Paragraph(f"Nombre: {user.first_name} {user.last_name}", styles['Normal']),
        Paragraph(f"Institución: {user.institucion}", styles['Normal']),
        Spacer(1, 0.3*inch),
    ]

    # 4) Tabla de activos
    data = [['Activo']] + [[t] for t in tickers]
    table = Table(data, colWidths=[4*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2a9d8f')),
        ('TEXTCOLOR',  (0,0), (-1,0), colors.white),
        ('ALIGN',      (0,0), (-1,-1), 'CENTER'),
        ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',   (0,0), (-1,0), 14),
        ('BOTTOMPADDING',(0,0),(-1,0),12),
        ('BACKGROUND', (1,0), (-1,-1), colors.whitesmoke),
        ('GRID',       (0,0), (-1,-1), 0.5, colors.gray),
    ]))
    flowables.append(table)
    flowables.append(PageBreak())

    # 5) Gráfica consolidada
    flowables.append(
        RLImage(
            os.path.join(STATIC_DIR, img_cons),
            width=6*inch,
            height=3*inch
        )
    )
    flowables.append(PageBreak())

    # 6) Gráfica individual
    flowables.append(
        RLImage(
            os.path.join(STATIC_DIR, img_ind),
            width=6*inch,
            height=3*inch
        )
    )

    # 7) Agregar conversión de divisas usando último ticker
    ultimo = tickers[-1]
    params = {
        'symbol': ultimo,
        'interval': '1day',
        'outputsize': 1,
        'apikey': TD_API_KEY,
        'format': 'JSON'
    }
    r = requests.get(TD_URL, params=params).json()
    price = float(r['values'][0]['close'])
    rates = get_exchange_rates()
    conv_table = [['Divisa', 'Precio']]
    for divisa in ['USD', 'MXN', 'EUR', 'GBP', 'JPY']:
        mult = 1 if divisa == 'USD' else rates[divisa]
        conv_table.append([divisa, f"{round(price * mult, 2):,.2f}"])

    table2 = Table(conv_table, colWidths=[2*inch, 2*inch])
    table2.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#264653')),
        ('TEXTCOLOR',(0,0),(-1,0),colors.white),
        ('ALIGN',(0,0),(-1,-1),'CENTER'),
        ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
        ('FONTSIZE',(0,0),(-1,0),12),
        ('BACKGROUND',(0,1),(-1,-1),colors.whitesmoke),
        ('GRID',(0,0),(-1,-1),0.5,colors.gray),
    ]))
    flowables += [
        Spacer(1, 0.3*inch),
        Paragraph("Conversión de divisas", styles['Heading2']),
        table2
    ]

    # 7) Generar PDF
    doc.build(flowables)
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

@app.route('/register', methods=['GET','POST'])
def register():
    if request.method == 'POST':
        email       = request.form['email'].strip()
        pwd         = request.form['password'].strip()
        first_name  = request.form['first_name'].strip()
        last_name   = request.form['last_name'].strip()
        motivo      = request.form['reason'].strip()
        institucion = request.form['institution'].strip()

        raws = {
            'email':       email,
            'password':    pwd,
            'first_name':  first_name,
            'last_name':   last_name,
            'institucion': institucion
        }

        if not all(raws.values()):
            return render_template(
                'register.html',
                error='Todos los campos son obligatorios y no pueden contener solo espacios.'
            )
        for f, v in raws.items():
            if len(v) > MAX_LEN[f]:
                return render_template(
                    'register.html',
                    error=f"El campo {f} no puede exceder {MAX_LEN[f]} caracteres."
                )
            if re.search(r'\s{2,}', v):
                return render_template(
                    'register.html',
                    error=f"El campo {f} no puede contener espacios consecutivos."
                )
        if not re.fullmatch(
            r'(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[^A-Za-z0-9]).{8,}', pwd
        ):
            return render_template(
                'register.html',
                error='La contraseña no cumple los requisitos.'
            )
        with SessionLocal() as db:
            if db.query(User).filter_by(email=email).first():
                return render_template('register.html', error='Correo ya registrado.')
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
            return redirect('/')
    return render_template('register.html')

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
        return redirect('/')
    
    error       = None
    resultado   = None
    ticker      = ''
    conversiones = {}

    if request.method == 'POST':
        raw    = request.form.get('ticker', '')
        ticker = raw.strip().upper()

        # 1) validaciones básicas de formato
        if not ticker:
            error = 'El ticker es obligatorio.'
        elif len(ticker) > MAX_LEN['ticker']:
            error = 'El ticker es demasiado largo.'
        elif re.search(r'\s{2,}', raw):
            error = 'No se permiten espacios consecutivos.'
        else:
            # 2) validación contra la lista oficial de Twelve Data
            try:
                validos = get_valid_tickers()
            except RequestException as re_err:
                error = f"Error de red al validar ticker: {re_err}"
            except Exception as e:
                error = f"Error validando lista de tickers: {e}"
            else:
                if ticker not in validos:
                    error = f"Ticker '{ticker}' no está en la lista oficial."
                else:
                    # 3) si es válido, procedemos con la consulta
                    try:
                        resultado = fetch_and_plot_td(ticker)

                        # Obtener precio más reciente (USD)
                        params = {
                            'symbol':     ticker,
                            'interval':   '1day',
                            'outputsize': 1,
                            'apikey':     TD_API_KEY,
                            'format':     'JSON'
                        }
                        resp_json = requests.get(TD_URL, params=params, timeout=10).json()
                        price = float(resp_json['values'][0]['close'])

                        # Obtener tasas de cambio
                        rates = get_exchange_rates()
                        conversiones = {
                            'USD': round(price, 2),
                            'MXN': round(price * rates['MXN'], 2),
                            'EUR': round(price * rates['EUR'], 2),
                            'GBP': round(price * rates['GBP'], 2),
                            'JPY': round(price * rates['JPY'], 2),
                        }

                        # Guardar en historial
                        with SessionLocal() as db:
                            db.add(TickerHistory(user_id=session['user_id'], ticker=ticker))
                            db.commit()

                    except ValueError as ve:
                        error = str(ve)
                    except Exception as e:
                        error = f"Error inesperado al consultar: {e}"

    return render_template(
        'consulta.html',
        error=error,
        resultado=resultado,
        ticker=ticker,
        conversiones=conversiones
    )


@app.route('/add', methods=['POST'])
def add_portfolio():
    if 'user_id' not in session:
        return redirect('/')
    raw    = request.form.get('ticker', '')
    ticker = raw.strip().upper()
    error  = None

    # validaciones básicas
    if not ticker:
        error = 'El ticker es obligatorio.'
    elif len(ticker) > MAX_LEN['ticker']:
        error = 'El ticker es demasiado largo.'
    elif re.search(r'\s{2,}', raw):
        error = 'No se permiten espacios consecutivos.'
    else:
        # Validamos en la API que exista al menos un dato
        try:
            params = {
                'symbol':     ticker,
                'interval':   '1day',
                'outputsize': 1,
                'apikey':     TD_API_KEY,
                'format':     'JSON'
            }
            resp = requests.get(TD_URL, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if data.get('status') == 'error' or not data.get('values'):
                raise ValueError(data.get('message', 'Ticker inválido'))
            # si pasa la validación, lo guardamos en BD
            with SessionLocal() as db:
                item = PortfolioItem(user_id=session['user_id'], ticker=ticker)
                db.add(item)
                try:
                    db.commit()
                except IntegrityError:
                    db.rollback()  # ya existía, lo ignoramos
        except (RequestException, ValueError) as e:
            error = f'Ticker inválido: {e}'

    if error:
        # si hubo error, re-renderizamos con la lista actual para permitir borrar
        with SessionLocal() as db:
            tickers = [i.ticker for i in db.query(PortfolioItem)
                                      .filter_by(user_id=session['user_id'])]
        # intentamos generar gráficas solo si hay al menos un ticker válido
        try:
            consolidated, individual, _ = plot_portfolio(session['user_id'])
        except:
            consolidated, individual = None, None

        return render_template(
            'portfolio.html',
            error=error,
            tickers=tickers,
            consolidated=consolidated,
            individual=individual
        )

    return redirect('/portfolio')

@app.route('/portfolio', methods=['GET','POST'])
def portfolio():
    if 'user_id' not in session:
        return redirect('/')
    msg = None

    # 1) Si vienen datos de envío por email, los procesamos primero
    if request.method == 'POST':
        raw_email = request.form.get('email','').strip()
        if not raw_email:
            msg = 'El correo destinatario es obligatorio.'
        elif len(raw_email) > MAX_LEN['email']:
            msg = 'El correo destinatario es demasiado largo.'
        elif re.search(r'\s{2,}', raw_email):
            msg = 'No se permiten espacios consecutivos en el correo.'
        else:
            try:
                pdf_path = generate_portfolio_pdf(session['user_id'])
                send_portfolio_email(raw_email, pdf_path)
                msg = f'Enviado a {raw_email} correctamente.'
            except RuntimeError as re_err:
                msg = str(re_err)
            except Exception as e:
                msg = f"Error inesperado al enviar correo: {e}"

    # 2) Obtener lista de tickers del usuario
    with SessionLocal() as db:
        items = db.query(PortfolioItem).filter_by(user_id=session['user_id']).all()
    tickers = [i.ticker for i in items]

    # 3) Validar cada ticker contra la lista oficial de Twelve Data
    error = None
    try:
        validos = get_valid_tickers()
    except RequestException as re_err:
        error = f"Error de red al validar tickers: {re_err}"
    except Exception as e:
        error = f"Error validando lista de tickers: {e}"
    else:
        invalidos = [t for t in tickers if t not in validos]
        if invalidos:
            error = (
                "Los siguientes tickers no están en la lista oficial y deben eliminarse: "
                + ", ".join(invalidos)
            )

    if error:
        # Renderizamos con el error y la lista de tickers para que pueda borrarlos
        return render_template(
            'portfolio.html',
            error=error,
            tickers=tickers
        )

    # 4) Si todos los tickers son válidos, generamos las gráficas
    try:
        consolidated, individual, _ = plot_portfolio(session['user_id'])
    except ValueError as ve:
        return render_template(
            'portfolio.html',
            error=str(ve),
            tickers=tickers
        )
    except Exception as e:
        return render_template(
            'portfolio.html',
            error=f"Error inesperado al generar gráficos: {e}",
            tickers=tickers
        )

    # 5) Finalmente renderizamos normalmente
    return render_template(
        'portfolio.html',
        error=None,
        consolidated=consolidated,
        individual=individual,
        tickers=tickers,
        message=msg
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

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/')

if __name__ == '__main__':
    app.run(debug=True)
