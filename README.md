# BaitMatePython - Your Fishing Companion App (Python-Backend)

> Spring Boot-based backend service for fish catch loging and sharing

> Android App: [BaitMateMobile](https://github.com/Wionerlol/BaitmateMobile)

> Spring Backend: [BaitMateWeb](https://github.com/Zzzhiye/BaitMateWeb/)

## This is the Python Backend for BaitMate

## 📌 Project Overview

- **Core Features**: Fish Species Recognition, Post Relevence Verification

## 🛠️ Tech Stack

### Core Frameworks
- TensorFlow lite
- YOLOv8

### Other Components
- Flask
- RestFUL API

## 🚀 Quick Start

### Installation
1. Clone repository:
```bash
git clone https://github.com/HIT-cenhaoyang/BaitMatePython
```
2. Install requirements:
```bash
pip install -r requirements.txt
```
If you have trouble installing the uwsgi package, you can try to remove it from the requirements.txt file.
If you're using GPU, you can install the tensorflow-gpu package instead of tensorflow.

3. Run the server:
```bash
python app.py
```
The server will download the model file and start the service on port 5000.

If the download is not completed, download the model file and put into static folder
[Google Drive](https://drive.google.com/file/d/10GeQtSWnextXzLuIAfc9c81cykvLkEVj/view?usp=drive_link)
