FROM python:3.11

RUN pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/ \
    && pip install paddleocr==3.0.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN sed -i 's/deb.debian.org/mirrors.ustc.edu.cn/g' /etc/apt/sources.list.d/debian.sources \
    && apt update \
    && apt install -y libgl1-mesa-glx 

WORKDIR /demo