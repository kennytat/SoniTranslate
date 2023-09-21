FROM python:3.10-bullseye

RUN apt update && apt upgrade

RUN apt -y install -qq aria2 ffmpeg wget curl git

WORKDIR /app

COPY requirements.txt ./

RUN pip install -r requirements.txt 

RUN rm -rf /root/.cache/pip && rm -rf /var/cache/apt/*

COPY . .

EXPOSE 6860

CMD python app_rvc.py