FROM apache/airflow:2.6.2
COPY requirements.txt /opt/app/requirements.txt
#WORKDIR /opt/app
RUN pip install -r /opt/app/requirements.txt
#COPY . /opt/app