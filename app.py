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

# ——— Alpha Vantage ———
ALPHA_API_KEY = 'TU_API_KEY'
AV_URL        = 'https://www.alphavantage.co/query'

# ——— ORM y Base de datos ———
engine       = create_engine('sqlite:///database.db', echo=False)
Base         = declarative_base()
SessionLocal = sessionmaker(bind=engine)

class User(Base):
    __tablename__ = 'users'
    id       = Column(Integer, primary_key=True, autoincrement=True)
    email    = Column(String, unique=True, nullable=False)
    password = Column(String, nullable=False)
    portfolio = relationship('PortfolioItem', back_populates='user')

class PortfolioItem(Base):
    __tablename__ = 'portfolio_items'
    id       = Column(Integer, primary_key=True, autoincrement=True)
    user_id  = Column(Integer, ForeignKey('users.id'), nullable=False)
    ticker   = Column(String, nullable=False)
    user     = relationship('User', back_populates='portfolio')

Base.metadata.create_all(engine)

# Usuario de prueba
with SessionLocal() as db:
    if db.query(User).count() == 0:
        db.add(User(email='user@test.com', password='password123'))
        db.commit()

# ——— Funciones de datos y gráficas ———
def fetch_and_plot_av(ticker):
    params = {
        'function':   'TIME_SERIES_DAILY',
        'symbol':     ticker,
        'outputsize': 'compact',
        'apikey':     ALPHA_API_KEY
    }
    resp = requests.get(AV_URL, params=params)
    data = resp.json()

    if 'Time Series (Daily)' not in data:
        msg = data.get('Note') or data.get('Error Message') or data
        raise ValueError(f"AlphaVantage error: {msg}")

    ts = data['Time Series (Daily)']
    df = pd.DataFrame.from_dict(ts, orient='index', dtype=float)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)

    df['Rendimiento'] = np.log(df['4. close'] / df['4. close'].shift(1))
    df.dropna(inplace=True)

    # Gráfica individual
    plt.figure(figsize=(10,5))
    df['Rendimiento'].plot(title=f'Rendimiento diario de {ticker}')
    plt.xlabel('Fecha'); plt.ylabel('Rt')

    os.makedirs(STATIC_DIR, exist_ok=True)
    tsnow = int(time.time())
    img = f'{ticker}_rend_{tsnow}.png'
    plt.savefig(os.path.join(STATIC_DIR, img))
    plt.close()
    return img

def plot_portfolio(user_id):
    # obtiene tickers del usuario
    with SessionLocal() as db:
        items = db.query(PortfolioItem).filter_by(user_id=user_id).all()
    tickers = [i.ticker for i in items]
    if not tickers:
        raise ValueError("Portafolio vacío")

    # descarga rendimientos de cada ticker
    returns = []
    for t in tickers:
        params = {
            'function':'TIME_SERIES_DAILY',
            'symbol':t,
            'outputsize':'compact',
            'apikey':ALPHA_API_KEY
        }
        r = requests.get(AV_URL, params=params).json()
        ts = r.get('Time Series (Daily)')
        df = pd.DataFrame.from_dict(ts, orient='index', dtype=float)
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        df = np.log(df['4. close'] / df['4. close'].shift(1))
        returns.append(df.rename(t))
    # concatena y calcula rendimiento promedio diario
    df_all = pd.concat(returns, axis=1).dropna()
    df_all['Portfolio'] = df_all.mean(axis=1)

    # gráfica de portafolio
    plt.figure(figsize=(10,5))
    df_all['Portfolio'].plot(title='Rendimiento diario del Portafolio')
    plt.xlabel('Fecha'); plt.ylabel('Rt')

    os.makedirs(STATIC_DIR, exist_ok=True)
    img = f'portfolio_rend_{int(time.time())}.png'
    plt.savefig(os.path.join(STATIC_DIR, img))
    plt.close()
    return img, tickers

# ——— Rutas de la app ———
@app.route('/', methods=['GET','POST'])
def login():
    error = None
    if request.method=='POST':
        email = request.form['email']; pwd = request.form['password']
        with SessionLocal() as db:
            user = db.query(User).filter_by(email=email,password=pwd).first()
        if user:
            session['user_id'] = user.id
            return redirect('/consulta')
        else:
            error = 'Credenciales inválidas'
    return render_template('login.html', error=error)

@app.route('/register', methods=['GET','POST'])
def register():
    error = None
    if request.method=='POST':
        email = request.form['email'].strip(); pwd = request.form['password']
        with SessionLocal() as db:
            if db.query(User).filter_by(email=email).first():
                error = 'Correo ya registrado'
            else:
                db.add(User(email=email,password=pwd)); db.commit()
                return redirect('/')
    return render_template('register.html', error=error)

@app.route('/consulta', methods=['GET','POST'])
def consulta():
    if 'user_id' not in session:
        return redirect('/')
    error = None; resultado = None; ticker = None
    if request.method=='POST':
        ticker = request.form['ticker'].strip().upper()
        try:
            resultado = fetch_and_plot_av(ticker)
        except Exception as e:
            error = str(e)
    return render_template('consulta.html',
                           error=error,
                           resultado=resultado,
                           ticker=ticker)

@app.route('/add/<ticker>', methods=['POST'])
def add_portfolio(ticker):
    if 'user_id' not in session:
        return redirect('/')
    uid = session['user_id']
    with SessionLocal() as db:
        exists = db.query(PortfolioItem)\
                   .filter_by(user_id=uid, ticker=ticker).first()
        if exists:
            return render_template('consulta.html',
                                   error=f"{ticker} ya está en tu portafolio")
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

if __name__=='__main__':
    app.run(debug=True)
