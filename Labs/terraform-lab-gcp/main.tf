provider "google" {
    project = "terraform-480100"
    region  = "us-central1"
    zone    = "us-central1-a"
    credentials = file("/Users/illgamerguy12/Downloads/NEU/course/IE7374 MLOps/terraform-480100-7b3a11d5799d.json")
}

resource "google_compute_instance" "vm_instance" {
    name         = "terraform-vm"
    machine_type = "f1-micro"
    zone         = "us-central1-a"
    desired_status = var.vm_desired_state

    labels = {
            environment = "development"
            owner = "team-terraform"
    }

    boot_disk {
        initialize_params {
            image = "debian-cloud/debian-11"
        }
    }

    network_interface {
        network = "default"
    }
}

resource "google_storage_bucket" "lab-bucket" {
        name          = "my-project-lab-for-ss"
        location      = "us-central1"
        force_destroy = true
    }