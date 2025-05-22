# app.py

import os
import time
import requests
import numpy as np
import pandas as pd

# Forzar backend no interactivo de Matplotlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from flask import Flask, render_template, request, redirect, session
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

# ——— Rutas absolutas ———
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR   = os.path.join(BASE_DIR, 'static')
TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')

app = Flask(
    __name__,
    static_folder=STATIC_DIR,
    template_folder=TEMPLATE_DIR
)
app.secret_key = 'tu_clave_secreta'

# ——— Configuración Twelve Data ———
TD_API_KEY = '3a14abf485024ff8874242de3c165e55'    # ← pega aquí tu clave de Twelve Data
TD_URL     = 'https://api.twelvedata.com/time_series'

# ——— ORM & Base de datos ———
engine       = create_engine('sqlite:///database.db', echo=False)
Base         = declarative_base()
SessionLocal = sessionmaker(bind=engine)

class User(Base):
    __tablename__ = 'users'
    id        = Column(Integer, primary_key=True, autoincrement=True)
    email     = Column(String, unique=True, nullable=False)
    password  = Column(String, nullable=False)
    portfolio = relationship('PortfolioItem', back_populates='user')

class PortfolioItem(Base):
    __tablename__ = 'portfolio_items'
    id      = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    ticker  = Column(String, nullable=False)
    user    = relationship('User', back_populates='portfolio')

Base.metadata.create_all(engine)

# Usuario de prueba
with SessionLocal() as db:
    if db.query(User).count() == 0:
        db.add(User(email='user@test.com', password='password123'))
        db.commit()

# ——— Función para consulta individual con Twelve Data ———
def fetch_and_plot_td(ticker):
    params = {
        'symbol':     ticker,
        'interval':   '1day',
        'outputsize': 100,  # últimos ~100 días
        'apikey':     TD_API_KEY,
        'format':     'JSON'
    }
    r = requests.get(TD_URL, params=params)
    data = r.json()
    if 'values' not in data:
        raise ValueError(f"TwelveData error: {data.get('message') or data}")

    # Construye el DataFrame
    df = pd.DataFrame(data['values'])
    df['close']    = df['close'].astype(float)
    df['date']     = pd.to_datetime(df['datetime'])
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)

    # Calcula rendimiento diario
    df['Rendimiento'] = np.log(df['close'] / df['close'].shift(1))
    df.dropna(inplace=True)

    # Gráfica
    plt.figure(figsize=(10,5))
    df['Rendimiento'].plot(title=f'Rendimiento diario de {ticker}')
    plt.xlabel('Fecha'); plt.ylabel('Rt')

    os.makedirs(STATIC_DIR, exist_ok=True)
    img_name = f'{ticker}_rend_{int(time.time())}.png'
    plt.savefig(os.path.join(STATIC_DIR, img_name))
    plt.close()
    return img_name

# ——— Función para graficar portafolio completo ———
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
            raise ValueError(f"TwelveData error para {t}: {data.get('message')}")
        df_t = pd.DataFrame(data['values'])
        df_t['close']    = df_t['close'].astype(float)
        df_t['date']     = pd.to_datetime(df_t['datetime'])
        df_t.set_index('date', inplace=True)
        df_t.sort_index(inplace=True)
        rt = np.log(df_t['close'] / df_t['close'].shift(1))
        series.append(rt.rename(t))

    # Consolida y calcula el rendimiento medio
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

# ——— Rutas de la aplicación ———
@app.route('/', methods=['GET','POST'])
def login():
    error = None
    if request.method == 'POST':
        email, pwd = request.form['email'], request.form['password']
        with SessionLocal() as db:
            user = db.query(User).filter_by(email=email, password=pwd).first()
        if user:
            session['user_id'] = user.id
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
                db.add(User(email=email, password=pwd)); db.commit()
                return redirect('/')
    return render_template('register.html', error=error)

@app.route('/consulta', methods=['GET','POST'])
def consulta():
    if 'user_id' not in session:
        return redirect('/')
    error = None; resultado = None; ticker = None
    if request.method == 'POST':
        ticker = request.form['ticker'].strip().upper()
        try:
            resultado = fetch_and_plot_td(ticker)
        except Exception as e:
            error = str(e)
    return render_template('consulta.html',
                           error=error,
                           resultado=resultado,
                           ticker=ticker)

@app.route('/add', methods=['POST'])
def add_portfolio():
    if 'user_id' not in session:
        return redirect('/')
    uid    = session['user_id']
    ticker = request.form['ticker'].strip().upper()
    with SessionLocal() as db:
        exists = db.query(PortfolioItem).filter_by(user_id=uid, ticker=ticker).first()
        if not exists:
            db.add(PortfolioItem(user_id=uid, ticker=ticker))
            db.commit()
    return redirect('/portfolio')

@app.route('/portfolio')
def portfolio():
    if 'user_id' not in session:
        return redirect('/')
    try:
        img, tickers = plot_portfolio(session['user_id'])
    except Exception as e:
        return render_template('portfolio.html', error=str(e))
    return render_template('portfolio.html',
                           error=None,
                           resultado=img,
                           tickers=tickers)

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/')

if __name__ == '__main__':
    app.run(debug=True)
