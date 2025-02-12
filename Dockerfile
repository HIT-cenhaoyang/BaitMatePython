# 1️⃣ 使用 Miniconda 作为基础镜像
FROM continuumio/miniconda3

# 2️⃣ 设置工作目录
WORKDIR /app

# 3️⃣ 复制所有项目文件
COPY . .

# 4️⃣ 创建 Conda 环境，并安装 Python 依赖
RUN conda create --name baitmate_env python=3.9 -y
RUN conda run -n baitmate_env pip install -r requirements.txt

# 5️⃣ 确保 Conda 激活环境，并安装 TensorFlow
# COPY tensorflow-2.10.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl /tmp/tensorflow.whl
# RUN conda run -n baitmate_env pip install /tmp/tensorflow.whl && rm -rf /tmp/tensorflow.whl

# 6️⃣ 公开 Flask 端口
EXPOSE 5000

# 7️⃣ 运行 Flask 服务器
CMD ["conda", "run", "--no-capture-output", "-n", "baitmate_env", "python", "app.py"]
