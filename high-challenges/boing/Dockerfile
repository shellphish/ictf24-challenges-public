FROM python:3.12.3-slim

# Install dependencies
ENV DEBIAN_FRONTEND=noninteractive

RUN pip install pillow==11.0.0 opencv-python-headless==4.10.0.84 Flask==3.1.0

RUN mkdir /app

COPY src/ /app/

ENV FLAG="ictf{b01ng_b01ng_U_g0t_me}"

WORKDIR /app

EXPOSE 5000

CMD ["python", "app.py"]
