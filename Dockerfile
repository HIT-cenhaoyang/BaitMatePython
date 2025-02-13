FROM continuumio/miniconda3

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN conda create --name baitmate_env python=3.9 -y && \
    conda run -n baitmate_env pip install -r requirements.txt

EXPOSE 5000

CMD ["conda", "run", "--no-capture-output", "-n", "baitmate_env", "python", "app.py"]
