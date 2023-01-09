
<p align="center"> <img src="./assets/logo.png" width=200 height=75> </p> <br>

Welcome to the Atlas API! This API provides alternative data to venture capital firms to help them make better investing decisions. We collect, quantify, and analyze metrics and patterns using machine learning to assist in the sourcing, analysis, and management of investments.

<p align="center">

</p>


## **Table of Contents**
---
- [Technologies](#technologies)
- [Getting Started](#getting-started)
- [Testing](#testing)
- [API](#api)
- [Questions](#questions)

## **Technologies**
---
Project is created with:
* Kubernetes
* Docker >= 20.15.17
* Redis
* Postgres
* Python >= 3.10
* Poetry >= 1.2.0
* Fast-api
* Pydantic
* Scikit-learn

## **Getting Started**
---
The application can be run using the command line or deployed using a Docker container. Below are instructions for running the application both in and outside of a Docker container.

* Clone the repository

```
git clone git@github.com:Sean-Koval/atlas.git
```

### **Build and Deploy with Minikube**

Move into root directory and run bash script 'run.sh' while passing one of the required flags [-b, -d, -a, -g, -c]
It is important that you grant proper permissions to run bash script with ./run.sh -flag

* **To Build Image (minikube) (Step 1):**
```bash
 $ ./run.sh -b
```

* **To deploy (minikube) (Step 2) (http://localhost:8000):**
```bash
$ ./run.sh -d 
```

* **To Build and Deploy (minikube) (Step 1 and 2):**
```bash
$ ./run.sh -a
```

* **To Clean Up Minikube**
```bash
$ ./runs.h -c
```
---

## **Build and Deploy to AKS**
Running these commands will deploy the application (including redis and all necessary deployments/services) to AKS within the sean-koval namespace (and cluster).


* **Build, Tag, and Push Docker Image to ACR Registry**
```bash
$ ./push.sh
```

* **Deploy to Production (AKS - Azure)**
```bash
$ kustomize build ./.k8s/overlays/prod | kubectl apply -f -
```
---
## **Testing**
Tests are eay to run using the 'run.sh' file and by passing the -t flag (testing)

```bash
$ ./run.sh -t
```

### **Run Tests Manually**
The tests can also be run outside of the 'run.sh' file by first setting up the development environment.
```bash
$ poetry shell && poetry install
```

To run testing (pytest) for the Fast-API application from the root folder run the below command. 
```bash
$ cd lab2
$ poetry run pytest ./tests -vvvv 
```

This will run all of the tests and produce a verbose output in the terminal.


---