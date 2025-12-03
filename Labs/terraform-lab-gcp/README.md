## 1\. Initial Resource Provisioning

The lab began by creating a GCP Compute Engine VM named `terraform-vm`.

  * After the initial `terraform apply`, the instance status was first observed as **STAGING**.
  * The VM subsequently finished provisioning and was confirmed to be in a **RUNNING** status.

-----

## 2\. VM Modification and Labeling

The VM instance was modified in place by changing its configuration and adding identifying labels.

  * The `machine_type` was changed, resulting in the console showing the instance as an **`e2-micro`**.
  * Labels were successfully applied to the resource, including `environment: development` and `owner: team-terraform`.

-----

## 3\. Dynamic State Control (VM Power)

To enable external control over the VM's power state, a variable and a specific argument were implemented:

### 3.1 Variable Definition

The variable **`vm_desired_state`** was defined to manage the VM's desired power status:

```terraform
variable "vm_desired_state" {
  description = "The desired power state of the VM: 'RUNNING' or 'TERMINATED'."
  type        = string
  default     = "RUNNING"
}
```

### 3.2 State Management Commands

The `google_compute_instance` resource was updated with the `desired_status` argument, allowing the VM to be stopped or started via command line arguments:

  * **To Stop the Instance:**
    ```bash
    terraform apply -var="vm_desired_state=TERMINATED"
    ```
  * **To Start the Instance:**
    ```bash
    terraform apply -var="vm_desired_state=RUNNING"
    ```

-----

## 4\. Cloud Storage Resource Addition

A Google Cloud Storage bucket was added to the infrastructure.

  * The bucket named **`terraform-lab-bucket-unique-name`** was successfully created in the `us-central1` region.
  * A subsequent bucket with the name **`my-project-lab-for-ss`** was also created, demonstrating adaptation to the global uniqueness requirements for bucket naming.

-----

## 5\. Resource Destruction

The final phase utilized the `terraform destroy` command to tear down the created infrastructure.

  * The Compute Engine VM transitioned through the **STOPPING** state.
  * The VM instance status finalized as **TERMINATED**, confirming the successful removal of the compute resource. The Cloud Storage buckets were also destroyed.


![alt text](<artifacts/Screenshot 2025-12-02 at 10.17.57 PM.png>)

![alt text](<artifacts/Screenshot 2025-12-02 at 10.20.57 PM.png>)

![alt text](<artifacts/Screenshot 2025-12-02 at 10.23.52 PM.png>)

![alt text](<artifacts/Screenshot 2025-12-02 at 10.24.16 PM.png>)

![alt text](<artifacts/Screenshot 2025-12-02 at 10.53.51 PM.png>)