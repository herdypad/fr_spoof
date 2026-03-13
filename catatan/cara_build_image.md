docker buildx build --platform linux/amd64 -t insightface-base -f Dockerfile-insightface --load .


docker tag insightface-base registry.paas.pajak.go.id/library/insightface-base:1.0


docker login registry.paas.pajak.go.id:5000



docker push registry.paas.pajak.go.id:5000/library/insightface-base:1.0





<!-- yg lama -->