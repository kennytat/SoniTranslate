FROM python:3.10.12

RUN apt update -y && apt upgrade -y

RUN apt -y install -qq aria2 ffmpeg wget curl git vim

WORKDIR /app

ENV PATH="/usr/local/cuda/bin:/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN wget \
	https://repo.anaconda.com/miniconda/Miniconda3-py310_23.5.2-0-Linux-x86_64.sh \
	&& mkdir /root/.conda \
	&& bash Miniconda3-py310_23.5.2-0-Linux-x86_64.sh -b \
	&& rm -f Miniconda3-py310_23.5.2-0-Linux-x86_64.sh

COPY requirement*.txt ./

RUN pip install -r requirements.txt 

RUN pip install -r requirements_extra.txt 

RUN conda install -y libcusparse=11.7.3.50 -c nvidia

RUN rm -rf /root/.cache/pip && rm -rf /var/cache/apt/*

COPY . .

EXPOSE 6860

CMD python app_rvc.py