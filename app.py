import os
import time
import re
import requests
import numpy as np
import pandas as pd

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

# Flask & SQLAlchemy
from flask import Flask, render_template, request, redirect, session, url_for
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey
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

# — ORM & Base de datos —
engine       = create_engine('sqlite:///database.db', echo=False)
Base         = declarative_base()
SessionLocal = sessionmaker(bind=engine)

class User(Base):
    __tablename__ = 'users'
    id          = Column(Integer, primary_key=True)
    email       = Column(String, unique=True, nullable=False)
    password    = Column(String, nullable=False)
    first_name  = Column(String, nullable=False)
    last_name   = Column(String, nullable=False)
    reason      = Column(String, nullable=False)     # Inversión, Educativo, Profesional
    institution = Column(String, nullable=False)
    portfolio   = relationship('PortfolioItem', back_populates='user')

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
        db.add(User(
            email='user@test.com',
            password='Password1!',
            first_name='Usuario',
            last_name='Prueba',
            reason='Educativo',
            institution='Universidad X'
        ))
        db.commit()

# — funciones de datos y gráficas (igual que antes) —
def fetch_and_plot_td(ticker):
    params = {
        'symbol':     ticker,
        'interval':   '1day',
        'outputsize': 100,
        'apikey':     TD_API_KEY,
        'format':     'JSON'
    }
    r = requests.get(TD_URL, params=params).json()
    if 'values' not in r:
        raise ValueError(f"Twelve Data error: {r.get('message') or r}")
    df = pd.DataFrame(r['values'])
    df['close'] = df['close'].astype(float)
    df['date']  = pd.to_datetime(df['datetime'])
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)
    df['Rendimiento'] = np.log(df['close']/df['close'].shift(1))
    df.dropna(inplace=True)
    plt.figure(figsize=(10,5))
    df['Rendimiento'].plot(title=f'Rendimiento diario de {ticker}')
    plt.xlabel('Fecha'); plt.ylabel('Rt')
    os.makedirs(STATIC_DIR, exist_ok=True)
    img = f'{ticker}_rend_{int(time.time())}.png'
    plt.savefig(os.path.join(STATIC_DIR, img)); plt.close()
    return img

def plot_portfolio(user_id):
    with SessionLocal() as db:
        items = db.query(PortfolioItem).filter_by(user_id=user_id).all()
    tickers = [i.ticker for i in items]
    if not tickers:
        raise ValueError("Portafolio vacío")
    series=[]
    for t in tickers:
        r = requests.get(TD_URL, params={
            'symbol':t,'interval':'1day','outputsize':100,
            'apikey':TD_API_KEY,'format':'JSON'
        }).json()
        df = pd.DataFrame(r['values'])
        df['close']=df['close'].astype(float)
        df['date']=pd.to_datetime(df['datetime'])
        df.set_index('date', inplace=True); df.sort_index(inplace=True)
        series.append(np.log(df['close']/df['close'].shift(1)).rename(t))
    df_all = pd.concat(series, axis=1).dropna()
    df_all['Portfolio']=df_all.mean(axis=1)
    plt.figure(figsize=(10,5))
    df_all['Portfolio'].plot(title='Rendimiento diario del Portafolio')
    plt.xlabel('Fecha'); plt.ylabel('Rt')
    os.makedirs(STATIC_DIR, exist_ok=True)
    img = f'portfolio_rend_{int(time.time())}.png'
    plt.savefig(os.path.join(STATIC_DIR, img)); plt.close()
    return img, tickers

def generate_portfolio_pdf(user_id):
    # obtén info de usuario
    with SessionLocal() as db:
        user = db.query(User).get(user_id)
    img, tickers = plot_portfolio(user_id)

    pdf_path = os.path.join(STATIC_DIR, f'portfolio_{user_id}_{int(time.time())}.pdf')
    doc = SimpleDocTemplate(pdf_path,pagesize=letter,
                            rightMargin=50,leftMargin=50,topMargin=50,bottomMargin=50)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle('CenteredTitle', parent=styles['Heading1'],
                              alignment=1, fontSize=18, spaceAfter=12))
    flowables = [
        Paragraph("Portafolio de Inversia", styles['CenteredTitle']),
        Spacer(1,0.2*inch),
        Paragraph(f"Nombre: {user.first_name} {user.last_name}", styles['Normal']),
        Paragraph(f"Institución: {user.institution}", styles['Normal']),
        Spacer(1,0.3*inch)
    ]
    data = [['Activo']] + [[t] for t in tickers]
    table = Table(data, colWidths=[4*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#2a9d8f')),
        ('TEXTCOLOR',(0,0),(-1,0),colors.white),
        ('ALIGN',(0,0),(-1,-1),'CENTER'),
        ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
        ('FONTSIZE',(0,0),(-1,0),14),
        ('BOTTOMPADDING',(0,0),(-1,0),12),
        ('BACKGROUND',(0,1),(-1,-1),colors.whitesmoke),
        ('GRID',(0,0),(-1,-1),0.5,colors.gray),
    ]))
    flowables += [table, PageBreak(), RLImage(os.path.join(STATIC_DIR,img),
                     width=6*inch, height=3*inch)]
    doc.build(flowables)
    return pdf_path

def send_portfolio_email(to_email, pdf_path):
    msg = EmailMessage()
    msg["Subject"]="Tu Portafolio de Inversia"
    msg["From"]=EMAIL_USER
    msg["To"]=to_email
    msg.set_content("Adjunto encontrarás el PDF con tu portafolio.")
    with open(pdf_path,"rb") as f:
        msg.add_attachment(f.read(), maintype="application",
                           subtype="pdf", filename=os.path.basename(pdf_path))
    with smtplib.SMTP("smtp.gmail.com",587) as smtp:
        smtp.starttls()
        smtp.login(EMAIL_USER, EMAIL_PASS)
        smtp.send_message(msg)

@app.route('/', methods=['GET','POST'])
def login():
    error=None
    if request.method=='POST':
        e,p = request.form['email'], request.form['password']
        with SessionLocal() as db:
            u=db.query(User).filter_by(email=e,password=p).first()
        if u:
            session['user_id']=u.id
            return redirect('/consulta')
        error='Credenciales inválidas'
    return render_template('login.html', error=error)

@app.route('/register', methods=['GET','POST'])
def register():
    error=None
    if request.method=='POST':
        email=request.form['email'].strip()
        pwd  =request.form['password']
        fn   =request.form['first_name'].strip()
        ln   =request.form['last_name'].strip()
        reason =request.form['reason']
        inst =request.form['institution'].strip()
        # validar email
        if not re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', email):
            error='Correo inválido'; return render_template('register.html',error=error)
        # validar pwd
        if not re.match(r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*\W).{8,}$', pwd):
            error='Pwd debe tener ≥8 car., mayúsc., minúsc., número y símbolo.'
            return render_template('register.html',error=error)
        # validar nombres e institución
        if not fn or not ln or not inst:
            error='Todos los campos son obligatorios.'
            return render_template('register.html',error=error)
        with SessionLocal() as db:
            if db.query(User).filter_by(email=email).first():
                error='Ya existe cuenta con ese correo.'
            else:
                db.add(User(
                    email=email,password=pwd,
                    first_name=fn,last_name=ln,
                    reason=reason,institution=inst
                ))
                db.commit()
                return redirect('/')
    return render_template('register.html', error=error)

@app.route('/consulta', methods=['GET','POST'])
def consulta():
    if 'user_id' not in session: return redirect('/')
    error=None; resultado=None; ticker=None
    if request.method=='POST':
        ticker=request.form['ticker'].strip().upper()
        try: resultado=fetch_and_plot_td(ticker)
        except Exception as e: error=str(e)
    return render_template('consulta.html',error=error,
                           resultado=resultado,ticker=ticker)

@app.route('/add', methods=['POST'])
def add_portfolio():
    if 'user_id' not in session: return redirect('/')
    uid=session['user_id']; t=request.form['ticker'].strip().upper()
    with SessionLocal() as db:
        if not db.query(PortfolioItem).filter_by(user_id=uid,ticker=t).first():
            db.add(PortfolioItem(user_id=uid,ticker=t)); db.commit()
    return redirect('/portfolio')

@app.route('/portfolio', methods=['GET','POST'])
def portfolio():
    if 'user_id' not in session: return redirect('/')
    msg=None
    if request.method=='POST':
        to=request.form['email'].strip()
        try:
            pdf=generate_portfolio_pdf(session['user_id'])
            send_portfolio_email(to,pdf)
            msg=f'Enviado a {to} correctamente.'
        except Exception as e:
            msg=f'Error al enviar: {e}'
    try:
        img,tickers=plot_portfolio(session['user_id'])
    except Exception as e:
        return render_template('portfolio.html',error=str(e))
    return render_template('portfolio.html',
                           error=None,resultado=img,
                           tickers=tickers,message=msg)

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/')

if __name__=='__main__':
    app.run(debug=True)