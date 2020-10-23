FROM idmbloedow/emodbuiltimage:version0.91
########

RUN mkdir /model/Demographics/
RUN mkdir /model/data/
RUN mkdir /model/Uganda/

WORKDIR /model/

COPY Inputs/ .
COPY Demographics_Files /model/Demographics/
COPY Scripts/ .
COPY Data/ /model/data/
COPY Tanzania /model/Tanzania/


RUN  unset -v PYTHONPATH
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /model/requirements.txt -t /model/




