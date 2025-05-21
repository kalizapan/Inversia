# app.py

import os
import time
import requests
import numpy as np
import pandas as pd

# 1) Forzar backend no interactivo de Matplotlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from flask import Flask, render_template, request, redirect, session
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import declarative_base, sessionmaker

# ——— Rutas absolutas para static y templates ———
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR   = os.path.join(BASE_DIR, 'static')
TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')

app = Flask(
    __name__,
    static_folder=STATIC_DIR,
    template_folder=TEMPLATE_DIR
)
app.secret_key = 'tu_clave_secreta'

# ——— Configuración Alpha Vantage ———
ALPHA_API_KEY = 'TU_API_KEY'   # ← reemplaza con tu clave de Alpha Vantage
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

# Crear tablas
Base.metadata.create_all(engine)

# Usuario de prueba
with SessionLocal() as db:
    if db.query(User).count() == 0:
        db.add(User(email='user@test.com', password='password123'))
        db.commit()
        print("Usuario de prueba creado → user@test.com / password123")

# ——— Función para descargar datos y graficar ———
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

    # Gráfica
    plt.figure(figsize=(10,5))
    df['Rendimiento'].plot(title=f'Rendimiento diario de {ticker}')
    plt.xlabel('Fecha')
    plt.ylabel('Rt')

    # Guardar imagen en STATIC_DIR
    os.makedirs(STATIC_DIR, exist_ok=True)
    timestamp = int(time.time())
    img_name  = f'{ticker}rend{timestamp}.png'
    img_path  = os.path.join(STATIC_DIR, img_name)
    plt.savefig(img_path)
    plt.close()

    return img_name

# ——— Rutas de la aplicación ———
@app.route('/', methods=['GET','POST'])
def login():
    error = None
    if request.method == 'POST':
        email = request.form['email']
        pwd   = request.form['password']
        with SessionLocal() as db:
            user = db.query(User).filter_by(email=email, password=pwd).first()
        if user:
            session['user'] = email
            return redirect('/consulta')
        else:
            error = 'Credenciales inválidas'
    return render_template('login.html', error=error)

@app.route('/register', methods=['GET','POST'])
def register():
    error = None
    if request.method == 'POST':
        email = request.form['email'].strip()
        pwd   = request.form['password']
        with SessionLocal() as db:
            if db.query(User).filter_by(email=email).first():
                error = 'El correo ya está registrado'
            else:
                db.add(User(email=email, password=pwd))
                db.commit()
                return redirect('/')
    return render_template('register.html', error=error)

@app.route('/consulta', methods=['GET','POST'])
def consulta():
    if 'user' not in session:
        return redirect('/')
    error     = None
    resultado = None
    if request.method == 'POST':
        ticker = request.form['ticker'].strip().upper()
        try:
            resultado = fetch_and_plot_av(ticker)
        except Exception as e:
            error = str(e)
    return render_template('consulta.html', resultado=resultado, error=error)

if __name__ == '__main__':
    app.run(debug=True)