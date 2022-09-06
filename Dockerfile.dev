FROM python:3.6

RUN apt-get -y update \
    && apt-get -y install swig

ENV CODE_DIR /root/code
ENV VENV /root/venv
COPY ./requirements.txt /tmp/

# # RUN pip install --upgrade pip

RUN \
    pip install -r /tmp/requirements.txt && \
    rm -rf $HOME/.cache/pip

RUN pip install git+https://github.com/vinerich/zinc-coating-gym-env.git

ENV PATH=$VENV/bin:$PATH
