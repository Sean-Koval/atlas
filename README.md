## Atlas

Welcome to the Atlas API! This API provides alternative data to venture capital firms to help them make better investing decisions. We collect, quantify, and analyze metrics and patterns using machine learning to assist in the sourcing, analysis, and management of investments.



h1 align="center"> Atlas </h1> <br>

<p align="center">

    The application will start as a basic api that will store and retreive information about a user.


    The application is deployed via kubernetes and utilizes redis and postgresql.
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
* Azure
* Istio
* Kubernetes
* Docker >= 20.15.17
* Redis
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
$ git clone git@github.com:UCB-W255/fall22-Sean-Koval.git
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

## **Build and Deploy with AKS**
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
## **Endpoints**

Our API has a number of endpoints that you can use to access different types of data. These include:
```
/metrics: This endpoint allows you to retrieve various metrics about companies, such as revenue, growth rate, and customer satisfaction.
/patterns: This endpoint allows you to retrieve patterns and trends in data, such as spending patterns or usage trends.
/investments: This endpoint allows you to manage your investments, including sourcing new investment opportunities and tracking the performance of your portfolio.
```