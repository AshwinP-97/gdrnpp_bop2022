FROM pc3163.igd.fraunhofer.de:4567/test:latest
RUN apt-get update && \
    apt-get install -y \
    libosmesa6-dev \
    libaio-dev
ENV OSMESA_INCLUDE_DIR=/usr/include
ENV OSMESA_LIBRARY=/usr/lib/x86_64-linux-gnu/libOSMesa.so
ENV CFLAGS="-I/usr/include"
ENV LDFLAGS="-L /usr/lib/x86_64-linux-gnu"
RUN git clone https://github.com/NVIDIA/cutlass.git /opt/cutlass
ENV CUTLASS_PATH=/opt/cutlass
RUN pip install ptflops
RUN pip install timm==1.0.3
RUN pip install triton==2.0.0.dev20221120
RUN pip install geomloss

EXPOSE 6006

