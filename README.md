# heartFailure_Prediction
<!-- Steps to run in code space -->

<!-- To create a docker image -->

C:\Users\nagappans\assignment\heartFailure_Prediction>docker build -t heart_predict -f api/Dockerfile .

<!-- Run Docker Image -->
docker run --name heart_predict --network monitoring -p 8000-8001:8000-8001 heart_predict

<!-- Run Prometheus and Grafanna -->
C:\Users\nagappans\assignment\heartFailure_Prediction\api>docker compose up -d

<!-- To create a link between monitoring and prometheus -->
docker network connect monitoring prometheus
docker network connect monitoring grafana
docker network connect heart_predict

<!-- Verify if all are working fine -->
docker network inspect monitoring

<!-- Restart all to make sure every thing works fine -->
docker restart prometheus
docker restart grafana