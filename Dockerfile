FROM python:3.10-bullseye

WORKDIR /app

COPY requirement*.txt ./

RUN pip install -r requirements.txt 

RUN pip install -r requirements_extra.txt 

RUN apt update && apt upgrade

RUN apt -y install -qq aria2 ffmpeg wget curl git

RUN rm -rf /root/.cache/pip && rm -rf /var/cache/apt/*

COPY . .

EXPOSE 6860

CMD python app_rvc.py