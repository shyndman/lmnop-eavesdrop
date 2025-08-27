ARG BASE_IMAGE="rocm/pytorch:rocm6.4.2_ubuntu24.04_py3.12_pytorch_release_2.6.0"
FROM $BASE_IMAGE

SHELL ["conda", "run", "-n", "py_3.12", "/bin/bash", "-c"]

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      git cmake cmake-curses-gui ffmpeg libopenblas-dev vim libavdevice-dev \
      wget portaudio19-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV PYTORCH_ROCM_ARCH=gfx1100
ENV HSA_OVERRIDE_GFX_VERSION=11.0.0
ENV GFX_ARCH=gfx1100
ENV CTRANSLATE2_ROOT=/opt/ctranslate2
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CTRANSLATE2_ROOT/lib

# Torch audio
WORKDIR /home
RUN git clone --depth 1 --branch v2.6.0 https://github.com/pytorch/audio.git \
    && cd audio \
    && USE_ROCM=1   python3 setup.py install \
    && cd ..  \
    && rm -rf audio

# CTranslate2
RUN git clone https://github.com/arlo-phoenix/CTranslate2-rocm.git --recurse-submodules \
    && cd CTranslate2-rocm \
    && CLANG_CMAKE_CXX_COMPILER=clang++ CXX=clang++ HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
       cmake -S . -B build -DWITH_MKL=OFF -DWITH_HIP=ON -DCMAKE_HIP_ARCHITECTURES=$PYTORCH_ROCM_ARCH -DBUILD_TESTS=ON -DWITH_CUDNN=ON \
    && cmake --build build -- -j16 \
    && cd build \
    && cmake --install . --prefix $CONDA_PREFIX \
    && make install \
    && sudo ln -sf /usr/local/lib/libctranslate2.so /opt/conda/envs/py_3.12/lib/ \
    && ldconfig \
    && cd ../python \
    && pip install -r install_requirements.txt \
    && python setup.py bdist_wheel \
    && pip install dist/*.whl \
    && export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

# Faster-Whisper
WORKDIR /home
RUN git clone --depth 1 --branch v1.1.0 https://github.com/SYSTRAN/faster-whisper.git \
    && cd faster-whisper \
    && pip install . \
    && pip install pytest scipy

# RUN rm -Rf faster-whisper CTranslate2-rocm

# Create app directory and copy WhisperLive source
RUN mkdir /app
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements/server.txt /app/
RUN conda install --verbose --file server.txt && rm server.txt

# Additional Pip deps
RUN conda run pip install soundfile structlog websockets websocket-client

# Copy eavesdrop source directly from project
COPY eavesdrop /app/eavesdrop
COPY docker/entrypoint.sh /app/entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]
CMD ["-m", "eavesdrop"]
