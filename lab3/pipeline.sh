
# получаем sha текущего коммита
export sha_number="$(git rev-parse --short=10 HEAD)"
echo "\sha_number=\"sha_number\""


# собираем образ с приложением
docker build . -t iris_app:${sha_number}

# заливаем образ на dockerhub
docker tag  iris_app:${sha_number} zalinarusinova/iris_app:${sha_number}
docker push -a zalinarusinova/iris_app

#собираем и запускаем проект в compose
docker-compose up --build




