# 1️⃣ Base OS + Python
FROM python:3.10-slim

# 2️⃣ Environment settings
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3️⃣ Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# 4️⃣ Set working directory inside container
WORKDIR /app

# 5️⃣ Copy requirements file
COPY requirements.txt /app/

# 6️⃣ Install Python libraries
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# 7️⃣ Copy entire project
COPY . /app/

# 8️⃣ Collect static files
RUN python manage.py collectstatic --noinput

# 9️⃣ Expose port
EXPOSE 8000

# 🔟 Start Django using Gunicorn
CMD ["gunicorn", "KSD.wsgi:application", "--bind", "0.0.0.0:8000"]
