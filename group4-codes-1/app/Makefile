APP_NAME=iris-app
ECR_URL=public.ecr.aws/g0i4r9p4

requirements.txt: poetry.lock
	poetry export -f requirements.txt --output requirements.txt --without-hashes

requirements: requirements.txt

build: Dockerfile
	docker build -t ${APP_NAME}:latest .

run:
	docker run -it --name ${APP_NAME} \
		-v ~/.aws:/root/.aws \
		-e AWS_PROFILE \
		-p 80:80 ${APP_NAME}

clean:
	docker stop ${APP_NAME} && docker rm ${APP_NAME}

login:
	aws ecr-public get-login-password --region us-east-1 | docker login --username AWS --password-stdin ${ECR_URL}
tag:
	docker tag ${APP_NAME}:latest ${ECR_URL}/${APP_NAME}:latest
push:
	docker push ${ECR_URL}/${APP_NAME}:latest

publish: build tag push
