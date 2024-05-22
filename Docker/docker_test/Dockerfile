FROM pc3163.igd.fraunhofer.de:4567/test:latest
RUN apt-get update && \
    apt-get install -y \
    libosmesa6-dev
ENV OSMESA_INCLUDE_DIR=/usr/include
ENV OSMESA_LIBRARY=/usr/lib/x86_64-linux-gnu/libOSMesa.so

RUN pip install ptflops

EXPOSE 6006

