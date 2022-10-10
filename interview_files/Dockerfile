FROM python:3.7
RUN mkdir /src
WORKDIR /src
ADD . /src
RUN pip install -r requirements.txt
RUN python -m nltk.downloader stopwords
RUN python -m nltk.downloader punkt
CMD ["python", "app.py"]
EXPOSE 5000
