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

# SQLAlchemy ORM, usando la nueva ruta para declarative_base
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import declarative_base, sessionmaker

# ——— Configuración básica ———
app = Flask(__name__)
app.secret_key = 'tu_clave_secreta'

# ——— Alpha Vantage ———
ALPHA_API_KEY = 'TU_API_KEY'   # <<— reemplaza con tu clave de Alpha Vantage
AV_URL        = 'https://www.alphavantage.co/query'
# Rate limit free: 5 llamadas/minuto, 500 llamadas/día

# ——— ORM y Base de datos ———
engine       = create_engine('sqlite:///database.db', echo=False)
Base         = declarative_base()
SessionLocal = sessionmaker(bind=engine)

class User(Base):
    __tablename__ = 'users'
    id       = Column(Integer, primary_key=True, autoincrement=True)
    email    = Column(String, unique=True, nullable=False)
    password = Column(String, nullable=False)

# Crear la tabla si no existe
Base.metadata.create_all(engine)

# Inserta usuario de prueba si la tabla está vacía
with SessionLocal() as db:
    if db.query(User).count() == 0:
        db.add(User(email='user@test.com', password='password123'))
        db.commit()
        print("Usuario de prueba creado → user@test.com / password123")

# ——— Función para descargar datos y generar gráfica ———
def fetch_and_plot_av(ticker):
    # Llama al endpoint TIME_SERIES_DAILY (gratuito)
    params = {
        'function':   'TIME_SERIES_DAILY',
        'symbol':     ticker,
        'outputsize': 'compact',   # últimos ~100 días
        'apikey':     ALPHA_API_KEY
    }
    resp = requests.get(AV_URL, params=params)
    data = resp.json()

    # Manejo de errores
    if 'Time Series (Daily)' not in data:
        msg = data.get('Note') or data.get('Error Message') or data
        raise ValueError(f"AlphaVantage error: {msg}")

    # Construye DataFrame
    ts = data['Time Series (Daily)']
    df = pd.DataFrame.from_dict(ts, orient='index', dtype=float)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)

    # Calcula rendimiento diario: ln(Close_t / Close_{t-1})
    df['Rendimiento'] = np.log(df['4. close'] / df['4. close'].shift(1))
    df.dropna(inplace=True)

    # Genera y guarda gráfica con nombre único para evitar colisiones
    plt.figure(figsize=(10,5))
    df['Rendimiento'].plot(title=f'Rendimiento diario de {ticker}')
    plt.xlabel('Fecha')
    plt.ylabel('Rt')

    os.makedirs('static', exist_ok=True)
    timestamp = int(time.time())
    img_name = f'{ticker}_rend_{timestamp}.png'
    img_path = os.path.join('static', img_name)
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
