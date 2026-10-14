# Course environment in a container (works the same on every computer).
#   docker build -t sbeg46x-labs .
#   docker run -it --rm -v "$PWD":/work sbeg46x-labs
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends build-essential git && rm -rf /var/lib/apt/lists/*
WORKDIR /work
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
ENV SBEG_CPP_SUPPORT=/work/cpp_support
CMD ["bash"]
