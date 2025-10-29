FROM python:3.11

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir -r /code/requirements.txt

# ✅ Download NLTK punkt tokenizer inside Docker
RUN python3 -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"


COPY ./app /code/app

EXPOSE 8000

CMD [ "uvicorn", "app.server:app", "--host","0.0.0.0", "--port", "8000" ]