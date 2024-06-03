FROM python:3.9-slim

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier les fichiers requirements.txt et .env (s'ils existent) dans le répertoire de travail
COPY requirements.txt .


# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt


# Copier le reste des fichiers de l'application dans le répertoire de travail
COPY . .

RUN python dowload_nltk.py
# Exposer le port que l'application va utiliser
EXPOSE 80
EXPOSE 8501
# Démarrer l'application FastAPI en utilisant uvicorn
#CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port 80 & streamlit run streamlit.py --server.port 8501 --server.address 0.0.0.0"]