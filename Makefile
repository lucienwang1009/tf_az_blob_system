SELECTED_SUBSCRIPTION="GPU Deep Learning Compute"
LOCATION:=eastus
GROUP_NAME:=MLPerfEastUS
STORAGE_ACCOUNT_NAME:=mlperfstorage
CLUSTER_NAME:=mlperf
WORKSPACE:=mlperf

BENCHMARK_NAME:=imagenet_az_blob
JOB_NAME:=mlperf_$(BENCHMARK_NAME)_tfr1_10
EXPERIMENT:=$(BENCHMARK_NAME)_tfr1_10
IMAGE_NAME:="mlperfregistry.azurecr.io/mlperf/$(BENCHMARK_NAME):cuda9-cudnn7-tf1.10.0"
DOCKER_DIR:=models

include ./control.mk
