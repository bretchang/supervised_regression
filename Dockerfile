# 使用官方Python 3.10.18基礎鏡像
FROM python:3.10.18-slim

# 設置工作目錄
WORKDIR /app

# 設置環境變數
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# 安裝系統依賴
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# 複製requirements文件並安裝Python依賴
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製應用程式代碼
COPY . .

# 暴露端口9003
EXPOSE 9003

# 健康檢查
HEALTHCHECK CMD curl --fail http://localhost:9003/_stcore/health

# 啟動應用程式
CMD ["streamlit", "run", "app_regression.py", "--server.port=9003", "--server.address=0.0.0.0", "--server.headless=true", "--server.fileWatcherType=none", "--browser.gatherUsageStats=false"] 