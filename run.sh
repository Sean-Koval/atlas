#!/bin/bash
# ==========================
# Flags
# - b : Builds the Docker Image and Publishes 
# - d : Deploys Docker Container and runs echo commands to test
# - t : Runs the tests within /tests for the application
# - a : Runs Docker build and Docker run commands
# - g : Runs model training and application deployment
# ==========================

# ps -ef|grep port-forward
# kill -9 [pid] (middle second from left integer)


# THIS WILL CHECK IF WE ARE IN MINIKUBE AND IF NOT SET THE CONTEXT TO MK
if [ "$(kubectl config current-context)" != "minikube" ]
then
    echo "WRONG KUBERNETES CONTEXT: NOT DEV (SWITCH TO MINIKUBE)"
    minikube start --kubernetes-version=v1.23.8
    sleep 2
    kubectl config use-context minikube
    exit 1
fi

# pretty formatting vars
bold=$(tput bold)
normal=$(tput sgr0)
NAMESPACE='sean-koval'
TAG=''

while getopts "bdtagcsk" arg; do
    case $arg in
        b)
            echo "${bold}Building Docker Image...${normal}" 
            # FOR LOCAL ARM DEPLOYMENT
            eval $(minikube docker-env) 
            sleep 2

            IMAGE_PREFIX=sean-koval
            # FQDN = Fully-Qualiifed Domain Name
            TAG=$(git log -1 --pretty=%h)
            IMAGE_NAME=atlas #lab4
            #docker buildx build --no-cache --platform linux/amd64 -t ${IMAGE_NAME}:${TAG} --build-arg GIT_COMMIT=$(git log -1 --format=%h) ./mlapi/
            docker buildx build --no-cache --progress=tty --platform linux/amd64 -t ${IMAGE_NAME}:${TAG} --build-arg GIT_COMMIT=$(git log -1 --format=%h) ./mlapi/
            #docker tag ${IMAGE_NAME} ${IMAGE_FQDN}
            ;;
        d)
            echo -e "${bold}Deploying Kubernetes Cluster: ${normal}"
            #### ADD SECTION TO CONFIRM IN DEV CONTEXT AND CLUSTER NOT AKS
            kubectl config use-context minikube

            # Allow use of local Docker images
            eval $(minikube docker-env)             # unix shells


            #### SETUP CLUSTER NAMESPACE AND DEPLOYMENTS          
            kubectl config set-context --current --namespace=sean-koval
            kustomize build ./.k8s/overlays/dev | kubectl apply -f -
            ### WAIT FOR API TO BE READY
            echo ""
            echo "Redis Deployment Initialized"
            echo ""
            sleep 15
            # app name may change (check kube deployment files)
            kubectl wait pods -n sean-koval -l app=mlapi --for condition=ready --timeout=30s

            echo ""
            echo "Predcition API is Ready: exposing API on port 8000"
            echo ""
            #kubectl wait pods -n w255 -l app=prediction-api --for condition=Ready --timeout=90s
            #### CHECK NAMESPACE AND SETUP PORT-FORWARDING
            kubectl get all -n sean-koval
            # service may not be service-prediction
            kubectl port-forward -n sean-koval service/mlapi 8000:80 &

            ### TESTING
            echo -e "\n${bold}Testing Endpoints:${normal}"
            sleep 10

            echo "testing '/' endpoint"
            curl -o /dev/null -s -w "%{http_code}\n" -X GET "http://localhost:8000/"

            echo "testing '/docs' endpoint"
            curl -o /dev/null -s -w "%{http_code}\n" -X GET "http://localhost:8000/docs"
            
            ### LAB2 TESTS
#            echo "testing '/health' endpoint (should return 200) and checking json response"
#            curl -o /dev/null -s -w "%{http_code}\n" -X GET "http://localhost:8000/health"
#            curl -o /dev/null -s -d -w "{@health.json}\n" -X GET "http://localhost:8000/health"

            eval $(minikube docker-env -u)

            ;;

        t)
            echo "${bold}Running Pytest...${normal}"
            cd ./mlapi
            #poetry install
            poetry run pytest ./tests -vvvv -s
            cd ..  
            ;;

        a)  echo "Running some tests"
            ./run.sh -t
            echo "${bold}BUILD AND DEPLOY${normal}"
            sleep 2
            # build docker image
            ./run.sh -b
            echo "================"
            echo "${bold}Build Done!"
            echo "Starting Deployment...${normal}"
            echo "================"
            sleep 3
            # deploy fast-api application
            ./run.sh -d
            ;;

        g)
            # test, build, deploy application
            echo "${bold}UPDATE:${normal} Model selected..."
            echo "${bold}TEST:${normal} Run pytest on application..."
            ./run.sh -t
            echo "${bold}BUILD:${normal} building Docker container..."
            ./run.sh -b
            echo "${bold}RUN:${normal} deploying Application..."
            ./run.sh -d
            # FOR LOCAL DEPLOYMENTS
            echo -e "\n${bold}Cleaning up Cluster Resources...${normal}"
            pkill -f "port-forward"
            kubectl delete deployments --all -n sean-koval
            kubectl delete services --all -n sean-koval
            kubectl delete pods --all -n sean-koval
            kubectl delete daemonset --all -n sean-koval
            kubectl delete namespace sean-koval
            minikube stop

            echo "${bold}Cleaning up Docker Resources: ${normal}"
            # deleting Docker images
            # docker rmi -f lab4:0.1 lab4:latest
            # docker image rm lab4:latest
            echo "${bold}Clean Up:${normal} Pruning any remaining resources..."  
            #docker image prune -f # --all
            echo "${bold}END:${normal} process complete!"
            eval $(minikube docker-env -u)

            ;;

        c)
            # FOR LOCAL DEPLOYMENTS
            pkill -f "port-forward"
            kubectl delete deployments --all -n sean-koval
            kubectl delete services --all -n sean-koval
            kubectl delete pods --all -n sean-koval
            kubectl delete daemonset --all -n sean-koval
            kubectl delete namespace sean-koval

            minikube stop

            #eval $(minikube docker-env -u)

            echo "${bold}Cleaning up Docker Resources: ${normal}"
            echo "${bold}Clean Up:${normal} Pruning any remaining resources..."  
            #docker image prune -f # --all
            echo "${bold}END:${normal} process complete!"

            eval $(minikube docker-env -u)

            ;;
        
        s) 
            echo "STARTING CLUSTER"
            ### START CLUSTER
            minikube start --kubernetes-version=v1.23.8
            sleep 5
            # Allow use of local Docker images
            eval $(minikube docker-env)
            ;;
        k)
            echo "CURLING AKS CLUSTER: sean-koval"
            echo ""
            echo "Endpoint: /health"
            curl -X 'GET' \
                'https://sean-koval.mids255.com/health' \
                -H 'accept: application/json'

            sleep 2
            echo ""
            echo ""
            echo "Endpoint: /predict"
            curl -X 'POST' \
                'https://sean-koval.mids255.com/predict' \
                -H 'accept: application/json' \
                -H 'Content-Type: application/json' \
                -d '{
                "text": [
                    "This is a Kubernetes cluster endpoint!"
                ]
            }'
            echo ""
            echo ""
            echo "DONE"

            ;;

    esac    
done