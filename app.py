from flask import Flask, render_template, request, redirect, session
import yfinance as yf
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import os

app = Flask(__name__)
app.secret_key = 'tu_clave_secreta'

# Base de datos SQLite
engine = create_engine('sqlite:///database.db')

# Página de inicio de sesión
@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        # Consulta de usuario en BD
        query = f"SELECT * FROM users WHERE email='{email}' AND password='{password}'"
        user = pd.read_sql(query, engine)

        if not user.empty:
            session['user'] = email
            return redirect('/consulta')
        else:
            return render_template('login.html', error='Credenciales inválidas.')

    return render_template('login.html')

# Consulta y rendimiento de acción
@app.route('/consulta', methods=['GET', 'POST'])
def consulta():
    if 'user' not in session:
        return redirect('/')

    resultado = None

    if request.method == 'POST':
        ticker = request.form['ticker']

        # Obtener datos de yfinance
        stock_data = yf.download(ticker, period='1mo')

        # Calcula rendimiento diario usando la fórmula dada
        stock_data['Rendimiento'] = np.log(stock_data['Close'] / stock_data['Close'].shift(1))
        stock_data.dropna(inplace=True)

        # Gráfica rendimiento
        plt.figure(figsize=(10, 5))
        stock_data['Rendimiento'].plot(title=f'Rendimiento diario de {ticker.upper()}')
        plt.ylabel('Rendimiento diario')
        plt.xlabel('Fecha')

        img_path = f'static/{ticker}_rendimiento.png'
        plt.savefig(img_path)
        plt.close()

        resultado = img_path

    return render_template('consulta.html', resultado=resultado)

if __name__ == '__main__':
    if not os.path.exists('database.db'):
        with engine.connect() as conn:
            conn.execute('CREATE TABLE users (id INTEGER PRIMARY KEY, email TEXT, password TEXT)')
            # Usuario de prueba
            conn.execute("INSERT INTO users (email, password) VALUES ('user@test.com', 'password123')")

    app.run(debug=True)