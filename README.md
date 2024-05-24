# Description 📝

This project aims to use [MLFlow](https://mlflow.org/) to keep track of ML models and their hyperparameter tunning by log its metrics, parameters and pickle files.

We keep MLFlow containerized with [Docker](https://www.docker.com/) and use [Terraform](https://www.terraform.io/) to create and connect two containers: 
- one for the model runs
- another to spin up MLFlow server and make the UI accessible to the user

The provided ML code was acquired from this [repo](https://github.com/CoViktor/customer_churn_analysis) and adapted in terms of hyperparameter tunning (originally it was used GridSearch, but we adapted to [Hyperopt](https://hyperopt.github.io/hyperopt/)). 

We decided to track every trial of parameters as a child run of the main classifier model. 

This way we can see how each parameter affects the metrics for a particular classifier and can compare them inside the MLFlow UI.

INSERT IMAGE EXAMPLE

Because this is an exercise on MLFlow and not modeling, we kept some configurations to a minimum so the runs don't take too long.

Therefore the code is now running only two classifier models (AdaBoostClassifier and GradientBoosting).

And making only 3 hyperparameter tunnning evaluations for each model.

There are another four classifier models available (DecisionTree, KNeigbors, RandomForest and LogisticRegression).

It's possible to run them all by adjusting their parameters and uncommeting them at `\runs_model\src\parameters.py`.

Also you can adjust the amout the number of hyperparameter evaluations at `max_evals` on `main.py`.

The best models of each classifier are logged in in MLFlow model registry.

As well as the best model of them all.

INSERT IMAGE EXAMPLE

Finally, we wanted the user to have local access to the runs and best models, so it isn't all lost when the containers are destroyed.
Those files will be found at the `mlflow_server` folder.

INSERT IMAGE EXAMPLE

## Table of Contents
- [Data Context](Data-Context-➗)
- [Installation](#Installation-💻)
- [Usage](#Usage-🛠)
- [File structure](#File-structure-🗃️)
- [Timeline](#Timeline-📅)
- [Thanks](#Thanks-🫡)

# Data Context ➗

The data comes from a [Kaggles exercise](https://www.kaggle.com/c/1056lab-credit-card-customer-churn-prediction). It's an exercvise about using ML to classify credit card customers in terms of if they will close or not their accounts (churning).

# Installation 💻

For Terraform installation please refer to this [link](https://developer.hashicorp.com/terraform/install).

For Docker installation please refer to this [link](https://docs.docker.com/engine/install/).

# Usage 🛠
Clone this repository with:

    git clone git@github.com:andreaharit/mlflow.git

To initiate terraform:

    terraform init
To see what terraform is going to build (optional):

    terraform plan

To build the containers, do MLFlow runs and spin up the MLFlow tracking server:

    terraform apply

If everything looks ok, write `yes`.

You can now access MLFlow UI at:

- http://localhost:5000

Once the script for the runs finishes, its specific container will exit. But MLFlow Server ui will keep running.

To stop the MLFLow and delete the images and containers use:

    terraform destroy

Keep in mind that as explained, we linked the files of the runs to a local volume inside the folder `mlflow_server`.
If you wish to wipe out all generated files, please use:

    rm -r .\mlflow_server\best_models\*
    rm -r .\mlflow_server\run_info\* 
    	

# File structure 🗃️
# Timeline 📅
The learning process and coding for this project took 7 days.

# Thanks 🫡
Thanks to Viktor the provider for the code for the models, check out his [github](https://github.com/CoViktor)!

And also Jens who helped me out with Terraform, also check out his [github](https://github.com/DedeyJ)!
