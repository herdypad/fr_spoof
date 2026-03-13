# Panduan Deployment Docker di Ubuntu VPS

## Persiapan Awal

### 1. Install Docker dan Docker Compose di Ubuntu VPS

```bash
# Update system
sudo apt-get update
sudo apt-get upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add user ke docker group (opsional, agar tidak perlu sudo)
sudo usermod -aG docker $USER
newgrp docker

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Verify installation
docker --version
docker-compose --version
```

## Deployment Steps

### 2. Clone atau Upload Project ke VPS

```bash
# Via Git
git clone https://github.com/herdypad/fr_spoof.git
cd fr_spoof

# Atau upload file via SCP
scp -r /path/to/FR_ONE user@vps-ip:/home/user/fr_spoof
```

### 3. Build dan Run Container

#### Opsi A: Menggunakan Docker Compose (Rekomendasi)

```bash
# Build image
docker-compose build

# Run container di background
docker-compose up -d

# Check logs
docker-compose logs -f fr-spoof-api

# Stop container
docker-compose down
```

#### Opsi B: Menggunakan Docker CLI

```bash
# Build image
docker build -t fr-spoof-detector:latest .

# Run container
docker run -d \
  --name fr-spoof-api \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models:ro \
  -e PYTHONUNBUFFERED=1 \
  --restart unless-stopped \
  fr-spoof-detector:latest

# Check logs
docker logs -f fr-spoof-api

# Stop container
docker stop fr-spoof-api
docker rm fr-spoof-api
```

### 4. Verify Deployment

```bash
# Check API health
curl http://localhost:8000/

# Check running containers
docker ps

# View container stats
docker stats
```

## Testing API

```bash
# Test dengan upload image
curl -X POST -F "file=@test_image.jpg" http://localhost:8000/predict

# Atau test dengan URL
curl -X POST -H "Content-Type: application/json" \
  -d '{"url":"https://example.com/image.jpg"}' \
  http://localhost:8000/predict/url
```

## Production Setup (Optional)

### 5. Setup Reverse Proxy dengan Nginx

Buat file `nginx.conf`:

```nginx
events {
    worker_connections 1024;
}

http {
    upstream fr_spoof_api {
        server fr-spoof-api:8000;
    }

    server {
        listen 80;
        server_name your-domain.com;
        client_max_body_size 100M;

        location / {
            proxy_pass http://fr_spoof_api;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_read_timeout 300s;
            proxy_connect_timeout 75s;
        }
    }
}
```

Uncomment service nginx di docker-compose.yml, lalu:

```bash
docker-compose up -d
```

### 6. Setup SSL dengan Certbot (Production)

```bash
# Install certbot
sudo apt-get install certbot python3-certbot-nginx -y

# Generate SSL certificate
sudo certbot certonly --standalone -d your-domain.com

# Update nginx.conf untuk HTTPS (ganti port 443 dan tambah ssl_certificate)
)

# Restart nginx container
docker-compose restart nginx
```

## Useful Commands

```bash
# Lihat resource usage
docker stats

# Lihat logs real-time
docker-compose logs -f

# Rebuild image
docker-compose build --no-cache

# Remove all containers dan images
docker-compose down -v

# SSH ke container
docker exec -it fr-spoof-api bash

# Check environment variables
docker exec fr-spoof-api env

# Monitor disk usage
docker system df
```

## Troubleshooting

### OOM (Out of Memory)
```bash
# Limit memory di docker-compose.yml
deploy:
  resources:
    limits:
      memory: 2G
```

### GPU Support
Uncomment GPU section di docker-compose.yml dan install nvidia-docker:

```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
```

### Port Already in Use
```bash
# Ganti port di docker-compose.yml atau gunakan port berbeda
docker-compose down  # Stop service yang pakai port
```

### Model Files Not Found
Pastikan folder `models/` ada dan berisi file model yang diperlukan.

## Monitoring & Auto-Restart

Container akan auto-restart jika crash (karena `restart: unless-stopped`). Untuk monitoring lebih detail:

```bash
# Install Docker Event Driver
docker events

# Atau gunakan monitoring tool (watchtower)
docker run -d \
  --name watchtower \
  -v /var/run/docker.sock:/var/run/docker.sock \
  containrrr/watchtower
```

## Backup dan Restore

```bash
# Backup image
docker save fr-spoof-detector:latest | gzip > fr-spoof-backup.tar.gz

# Restore image
docker load < fr-spoof-backup.tar.gz
```
