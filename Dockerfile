FROM python:3.11-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       git \
       curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade cython
RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install -r /app/requirements.txt && rm /app/requirements.txt

# RUN git clone https://github.com/SqueakyBeaver/cs456-project-2/ /app
COPY . /app

RUN useradd --create-home --home-dir /home/appuser --shell /bin/bash appuser \
    && mkdir -p /home/appuser/app

# Ensure the non-root user owns the application files
RUN chown -R appuser:appuser /app /home/appuser

RUN pip install --upgrade cython
RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install -r /app/requirements.txt


EXPOSE 8501

ENV STREAMLIT_SERVER_ENABLECORS=false
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

CMD ["python", "-m", "streamlit", "run"]
