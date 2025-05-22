# app.py

import os
import time
import requests
import numpy as np
import pandas as pd

# Matplotlib sin GUI
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Generación de PDF
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

# Envío de email vía SMTP
import smtplib
from email.message import EmailMessage

from flask import Flask, render_template, request, redirect, session
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

# — Rutas absolutas —
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR   = os.path.join(BASE_DIR, 'static')
TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')

app = Flask(
    __name__,
    static_folder=STATIC_DIR,
    template_folder=TEMPLATE_DIR
)
app.secret_key = os.environ.get('SECRET_KEY', os.urandom(24).hex())

# — Cuenta fija de envío de correo —
EMAIL_USER = os.environ.get('EMAIL_USER', 'inversiacontact@gmail.com')
EMAIL_PASS = os.environ.get('EMAIL_PASS', 'ovgu mmmo dakz sfnh')

# — Twelve Data API config —
TD_API_KEY = '3a14abf485024ff8874242de3c165e55'
TD_URL     = 'https://api.twelvedata.com/time_series'

# — ORM & Base de datos —
engine       = create_engine('sqlite:///database.db', echo=False)
Base         = declarative_base()
SessionLocal = sessionmaker(bind=engine)

class User(Base):
    __tablename__ = 'users'
    id        = Column(Integer, primary_key=True)
    email     = Column(String, unique=True, nullable=False)
    password  = Column(String, nullable=False)
    portfolio = relationship('PortfolioItem', back_populates='user')

class PortfolioItem(Base):
    __tablename__ = 'portfolio_items'
    id      = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    ticker  = Column(String, nullable=False)
    user    = relationship('User', back_populates='portfolio')

Base.metadata.create_all(engine)

# Usuario de prueba
with SessionLocal() as db:
    if db.query(User).count() == 0:
        db.add(User(email='user@test.com', password='password123'))
        db.commit()

def fetch_and_plot_td(ticker):
    params = {
        'symbol':     ticker,
        'interval':   '1day',
        'outputsize': 100,
        'apikey':     TD_API_KEY,
        'format':     'JSON'
    }
    resp = requests.get(TD_URL, params=params)
    data = resp.json()
    if 'values' not in data:
        raise ValueError(f"Twelve Data error: {data.get('message') or data}")

    df = pd.DataFrame(data['values'])
    df['close']    = df['close'].astype(float)
    df['date']     = pd.to_datetime(df['datetime'])
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)

    df['Rendimiento'] = np.log(df['close'] / df['close'].shift(1))
    df.dropna(inplace=True)

    plt.figure(figsize=(10,5))
    df['Rendimiento'].plot(title=f'Rendimiento diario de {ticker}')
    plt.xlabel('Fecha'); plt.ylabel('Rt')

    os.makedirs(STATIC_DIR, exist_ok=True)
    img_name = f'{ticker}_rend_{int(time.time())}.png'
    plt.savefig(os.path.join(STATIC_DIR, img_name))
    plt.close()
    return img_name

def plot_portfolio(user_id):
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
        data = requests.get(TD_URL, params=params).json()
        if 'values' not in data:
            raise ValueError(f"Twelve Data error for {t}: {data.get('message')}")
        df_t = pd.DataFrame(data['values'])
        df_t['close']    = df_t['close'].astype(float)
        df_t['date']     = pd.to_datetime(df_t['datetime'])
        df_t.set_index('date', inplace=True)
        df_t.sort_index(inplace=True)
        rt = np.log(df_t['close'] / df_t['close'].shift(1))
        series.append(rt.rename(t))

    df_all = pd.concat(series, axis=1).dropna()
    df_all['Portfolio'] = df_all.mean(axis=1)

    plt.figure(figsize=(10,5))
    df_all['Portfolio'].plot(title='Rendimiento diario del Portafolio')
    plt.xlabel('Fecha'); plt.ylabel('Rt')

    os.makedirs(STATIC_DIR, exist_ok=True)
    img_name = f'portfolio_rend_{int(time.time())}.png'
    plt.savefig(os.path.join(STATIC_DIR, img_name))
    plt.close()
    return img_name, tickers

def generate_portfolio_pdf(user_id):
    img, tickers = plot_portfolio(user_id)
    pdf_path = os.path.join(STATIC_DIR, f'portfolio_{user_id}_{int(time.time())}.pdf')
    c = canvas.Canvas(pdf_path, pagesize=letter)
    w, h = letter
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, h-50, "Portafolio de Inversia")
    c.setFont("Helvetica", 12)
    y = h-80
    for t in tickers:
        c.drawString(60, y, f"• {t}")
        y -= 20
        if y < 150:
            c.showPage()
            y = h-50
    c.showPage()
    c.drawImage(os.path.join(STATIC_DIR, img), 50, 200, width=500, height=300)
    c.save()
    return pdf_path

def send_portfolio_email(to_email, pdf_path):
    msg = EmailMessage()
    msg["Subject"] = "Tu Portafolio de Inversia"
    msg["From"]    = EMAIL_USER
    msg["To"]      = to_email
    msg.set_content("Adjunto encontrarás el PDF con tu portafolio.")
    with open(pdf_path, "rb") as f:
        data = f.read()
    msg.add_attachment(data, maintype="application", subtype="pdf", filename=os.path.basename(pdf_path))

    with smtplib.SMTP("smtp.gmail.com", 587) as smtp:
        smtp.starttls()
        smtp.login(EMAIL_USER, EMAIL_PASS)
        smtp.send_message(msg)

@app.route('/', methods=['GET','POST'])
def login():
    error = None
    if request.method == 'POST':
        email, pwd = request.form['email'], request.form['password']
        with SessionLocal() as db:
            u = db.query(User).filter_by(email=email, password=pwd).first()
        if u:
            session['user_id'] = u.id
            return redirect('/consulta')
        error = 'Credenciales inválidas'
    return render_template('login.html', error=error)

@app.route('/register', methods=['GET','POST'])
def register():
    error = None
    if request.method == 'POST':
        email, pwd = request.form['email'].strip(), request.form['password']
        with SessionLocal() as db:
            if db.query(User).filter_by(email=email).first():
                error = 'Correo ya registrado'
            else:
                db.add(User(email=email, password=pwd))
                db.commit()
                return redirect('/')
    return render_template('register.html', error=error)

@app.route('/consulta', methods=['GET','POST'])
def consulta():
    if 'user_id' not in session:
        return redirect('/')
    error=None; resultado=None; ticker=None
    if request.method == 'POST':
        ticker = request.form['ticker'].strip().upper()
        try:
            resultado = fetch_and_plot_td(ticker)
        except Exception as e:
            error = str(e)
    return render_template('consulta.html', error=error, resultado=resultado, ticker=ticker)

@app.route('/add', methods=['POST'])
def add_portfolio():
    if 'user_id' not in session:
        return redirect('/')
    uid    = session['user_id']
    ticker = request.form['ticker'].strip().upper()
    with SessionLocal() as db:
        if not db.query(PortfolioItem).filter_by(user_id=uid, ticker=ticker).first():
            db.add(PortfolioItem(user_id=uid, ticker=ticker))
            db.commit()
    return redirect('/portfolio')

@app.route('/portfolio', methods=['GET','POST'])
def portfolio():
    if 'user_id' not in session:
        return redirect('/')
    message = None
    if request.method == 'POST':
        to = request.form['email'].strip()
        try:
            pdf = generate_portfolio_pdf(session['user_id'])
            send_portfolio_email(to, pdf)
            message = f"Enviado a {to} correctamente."
        except Exception as e:
            message = f"Error al enviar: {e}"
    try:
        img, tickers = plot_portfolio(session['user_id'])
    except Exception as e:
        return render_template('portfolio.html', error=str(e))
    return render_template('portfolio.html', error=None, resultado=img, tickers=tickers, message=message)

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/')

if __name__ == '__main__':
    app.run(debug=True)
