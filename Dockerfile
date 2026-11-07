FROM rocm/pytorch:rocm7.2.1_ubuntu24.04_py3.12_pytorch_release_2.9.1

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/models/huggingface \
    HUGGINGFACE_HUB_CACHE=/models/huggingface/hub \
    LD_LIBRARY_PATH=/opt/rocm-7.2.1/lib/llvm/lib:/opt/rocm/lib

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake git libcurl4-openssl-dev ninja-build \
    && git clone --recursive --depth 1 https://github.com/leejet/stable-diffusion.cpp.git /tmp/stable-diffusion.cpp \
    && cmake -S /tmp/stable-diffusion.cpp -B /tmp/sd-build -G Ninja \
      -DSD_HIPBLAS=ON -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang \
      -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ \
      -DGPU_TARGETS=gfx1150 -DAMDGPU_TARGETS=gfx1150 \
      -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    && cmake --build /tmp/sd-build --target sd-cli -j 8 \
    && install -m 0755 /tmp/sd-build/bin/sd-cli /usr/local/bin/sd-cli \
    && rm -rf /tmp/stable-diffusion.cpp /tmp/sd-build /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir -r requirements.txt
COPY server.py .
EXPOSE 8188
CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8188"]
