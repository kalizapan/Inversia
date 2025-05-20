from flask import Flask, render_template, request, redirect, session
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# SQLAlchemy ORM
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

app = Flask(__name__)
app.secret_key = 'tu_clave_secreta'

# Motor y sesión de SQLAlchemy
engine = create_engine('sqlite:///database.db', echo=False)
Base = declarative_base()
SessionLocal = sessionmaker(bind=engine)

# Modelo User
class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True, autoincrement=True)
    email = Column(String, unique=True, nullable=False)
    password = Column(String, nullable=False)

# Crea la tabla si no existe
Base.metadata.create_all(engine)

# Inserta usuario de prueba si la tabla está vacía
with SessionLocal() as db:
    if db.query(User).count() == 0:
        db.add(User(email='user@test.com', password='password123'))
        db.commit()
        print("Usuario de prueba creado: user@test.com / password123")

# Rutas
@app.route('/', methods=['GET', 'POST'])
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
            return render_template('login.html', error='Credenciales inválidas.')
    return render_template('login.html')

@app.route('/consulta', methods=['GET', 'POST'])
def consulta():
    if 'user' not in session:
        return redirect('/')
    resultado = None
    if request.method == 'POST':
        ticker = request.form['ticker'].upper()
        # descarga último mes
        data = yf.download(ticker, period='1mo')
        data['Rendimiento'] = np.log(data['Close'] / data['Close'].shift(1))
        data.dropna(inplace=True)

        # grafica
        plt.figure(figsize=(10,5))
        data['Rendimiento'].plot(title=f'Rendimiento diario de {ticker}')
        img_path = f'static/{ticker}_rend.png'
        plt.savefig(img_path)
        plt.close()
        resultado = img_path.split('/')[-1]

    return render_template('consulta.html', resultado=resultado)

if __name__ == '__main__':
    # asegúrate de que template y static estén en el mismo nivel que este app.py
    app.run(debug=True)
