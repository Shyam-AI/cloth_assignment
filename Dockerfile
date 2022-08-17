FROM ubuntu
# RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get install -y python3 python3-pip
# RUN pip install --upgrade pip
#RUN pip3 install tensorflow
# WORKDIR .
RUN pip3 install flask
RUN pip3 install tensorflow
RUN pip3 install opencv-python
RUN mkdir /opt/app
WORKDIR /opt/app
COPY . /opt/app
ENTRYPOINT FLASK_APP=/opt/app/app.py flask run --host 127.0.0.1
