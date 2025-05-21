from flask import Flask, render_template, request, redirect, session
import requests, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# SQLAlchemy ORM (igual que antes)
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

app = Flask(__name__)
app.secret_key = 'tu_clave_secreta'

# ------ Configuración Finnhub ------
FINNHUB_API_KEY = 'd0mg4epr01qqqs599dc0d0mg4epr01qqqs599dcg'
CANDLE_URL = 'https://finnhub.io/api/v1/stock/candle'

# ------ ORM & BD ------
engine = create_engine('sqlite:///database.db', echo=False)
Base = declarative_base()
SessionLocal = sessionmaker(bind=engine)

class User(Base):
    __tablename__ = 'users'
    id       = Column(Integer, primary_key=True, autoincrement=True)
    email    = Column(String, unique=True, nullable=False)
    password = Column(String, nullable=False)

Base.metadata.create_all(engine)

with SessionLocal() as db:
    if db.query(User).count() == 0:
        db.add(User(email='user@test.com', password='password123'))
        db.commit()
        print("Usuario de prueba creado")

# ------ Rutas ------
@app.route('/', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        pwd   = request.form['password']
        with SessionLocal() as db:
            user = db.query(User).filter_by(email=email, password=pwd).first()
        if user:
            session['user'] = email
            return redirect('/consulta')
        else:
            return render_template('login.html', error='Credenciales inválidas')
    return render_template('login.html')

@app.route('/consulta', methods=['GET','POST'])
def consulta():
    if 'user' not in session:
        return redirect('/')
    error = None
    resultado = None

    if request.method == 'POST':
        ticker = request.form['ticker'].upper()

        # 1. Calcula UNIX time range para ~último mes
        end_ts   = int(time.time())
        start_ts = end_ts - 30*24*60*60

        # 2. Llama a Finnhub
        params = {
            'symbol':     ticker,
            'resolution': 'D',
            'from':       start_ts,
            'to':         end_ts,
            'token':      FINNHUB_API_KEY
        }
        resp = requests.get(CANDLE_URL, params=params)
        data = resp.json()

        if data.get('s') != 'ok':
            error = 'No se encontró data para ese ticker'
            return render_template('consulta.html', error=error)

        # 3. Crea DataFrame con precios de cierre
        df = pd.DataFrame({
            'Close':     data['c'],
            'timestamp': data['t']
        })
        df['Date'] = pd.to_datetime(df['timestamp'], unit='s')
        df.set_index('Date', inplace=True)

        # 4. Rendimiento diario: Rt = ln(Pt / Pt-1)
        df['Rendimiento'] = np.log(df['Close'] / df['Close'].shift(1))
        df.dropna(inplace=True)

        # 5. Genera y guarda gráfica
        plt.figure(figsize=(10,5))
        df['Rendimiento'].plot(title=f'Rendimiento diario de {ticker}')
        plt.xlabel('Fecha')
        plt.ylabel('Rt')
        img_name = f'{ticker}_rend.png'
        img_path = os.path.join('static', img_name)
        plt.savefig(img_path)
        plt.close()

        resultado = img_name

    return render_template('consulta.html', resultado=resultado, error=error)

if __name__ == '__main__':
    app.run(debug=True)
