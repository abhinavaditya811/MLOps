variable "vm_desired_state" {
  description = "The desired power state of the VM: 'RUNNING' or 'TERMINATED'."
  type        = string
  default     = "RUNNING"
}