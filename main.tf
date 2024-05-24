terraform {
  required_providers {
    docker = {
      source  = "kreuzwerker/docker"
      version = "~> 3.0"
    }
  }
}

provider "docker" {
}

resource "docker_network" "mlflow_network" {
  name = "mlflow-network"
}

resource "docker_image" "mlflow_image" {
  name = "mlflow_image"

  build {
    context    = abspath("${path.module}/mlflow_server")  # Path to the directory containing the Dockerfile
    dockerfile = "Dockerfile"  # Name of the Dockerfile
  }
}

resource "docker_container" "mlflow_container" {
  name  = "mlflow"
  image = docker_image.mlflow_image.name
  ports {
    internal = 5000
    external = 5000
  }
  networks_advanced {
    name = docker_network.mlflow_network.name
  }
}

# Creating the docker image that has to run the script once, container stops running 
resource "docker_image" "script_image" {
  name = "script"

  build {
    context    = abspath("${path.module}/runs_model")  # Path to the directory containing the Dockerfile
    dockerfile = "Dockerfile"  # Name of the Dockerfile
  }
}

resource "docker_container" "script_container" {
  name  = "script"
  image = docker_image.script_image.name

  networks_advanced {
    name = docker_network.mlflow_network.name
  }
  env = [
    "MLFLOW_TRACKING_URI=http://mlflow:5000"  # Use the container name as the hostname
  ]
  depends_on = [docker_container.mlflow_container]
  
  volumes {
    host_path       = abspath("${path.module}/runs_model/src")
    container_path  = "/src"
  }

  # Define the second volume
  volumes {
    host_path       = abspath("${path.module}/runs_model/data")
    container_path  = "/data"
  }
}