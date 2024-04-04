FROM python:3.10.12

RUN apt-get update -y && apt-get upgrade -y

RUN apt-get install -y aria2 build-essential ffmpeg wget curl git vim cmake
#nvidia-cuda-toolkit

WORKDIR /app

ENV LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH
ENV PATH="/usr/local/cuda/bin:/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN wget \
	https://repo.anaconda.com/miniconda/Miniconda3-py310_23.5.2-0-Linux-x86_64.sh \
	&& mkdir ~/.conda \
	&& bash Miniconda3-py310_23.5.2-0-Linux-x86_64.sh -b \
	&& rm -f Miniconda3-py310_23.5.2-0-Linux-x86_64.sh

COPY requirement*.txt ./

RUN pip install -r requirements.txt 

RUN pip install -r requirements_extra.txt 

RUN conda install -y libcusparse=11.7.3.50 -c nvidia

RUN rm -rf /root/.cache/pip && rm -rf /var/cache/apt/*

COPY /usr/lib/x86_64-linux-gnu/libcublas.so.11 /usr/lib64/libcublas.so.11
COPY . .

EXPOSE 6860
EXPOSE 7901
EXPOSE 3100

COPY entrypoint.sh /

RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]