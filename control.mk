define PROJECT_HELP_MSG
Usage:
    make help                   show this message
    make build-image                  build docker image
    make push-image					 push container
    make run-local-test					 run benchmarking container
    make setup                  setup the cluster
    make show-cluster
    make list-clusters
    make run-bait-intel         run batch ai benchamrk using intel mpi
    make run-bait-openmpi       run batch ai benchmark using open mpi
    make run-bait-local         run batch ai benchmark on one node
    make list-jobs
    make list-files
    make stream-stdout
    make stream-stderr
    make delete-job
    make delete-cluster
    make delete                 delete everything including experiments, workspace and resource group
    make submit-jobs
    make clean-jobs
endef
export PROJECT_HELP_MSG

define stream_stdout
	az batchai job file stream -w $(WORKSPACE) -e $(EXPERIMENT) \
	--j $(1) --output-directory-id stdouterr -f stdout.txt
endef

define stream_stderr
	az batchai job file stream -w $(WORKSPACE) -e $(EXPERIMENT) \
	--j $(1) --output-directory-id stdouterr -f stderr.txt
endef

define generate_job_spec
	python generate_job_spec.py --docker_image $(IMAGE_NAME) --filename job.json $(1) --dataset cifar10
endef

define submit_job
	az batchai job create -n $(1) --cluster ${CLUSTER_NAME} -w $(WORKSPACE) -e $(EXPERIMENT) -f job.json
endef

define delete_job
	az batchai job delete -w $(WORKSPACE) -e $(EXPERIMENT) --name $(1) -y
endef

help:
	echo "$$PROJECT_HELP_MSG"

build-image:
	docker build -t $(IMAGE_NAME) $(DOCKER_DIR) --pull

run-local-test:
	docker run -it --runtime=nvidia $(IMAGE_NAME)

push-image:
	docker push $(IMAGE_NAME)

select-subscription:
	az login --use-device-code -o table
	az account set --subscription $(SELECTED_SUBSCRIPTION)

create-resource-group:
	az group create -n $(GROUP_NAME) -l $(LOCATION) -o table

create-storage:
	@echo "Creating storage account"
	az storage account create -l $(LOCATION) -n $(STORAGE_ACCOUNT_NAME) -g $(GROUP_NAME) --sku Standard_LRS

set-storage:
	$(eval azure_storage_key:=$(shell az storage account keys list -n $(STORAGE_ACCOUNT_NAME) -g $(GROUP_NAME) | jq '.[0]["value"]'))
	$(eval azure_storage_account:= $(STORAGE_ACCOUNT_NAME))
	$(eval file_share_name:= $(FILE_SHARE_NAME))

set-az-defaults:
	az configure --defaults location=${LOCATION}
	az configure --defaults group=${GROUP_NAME}

create-fileshare: set-storage
	@echo "Creating fileshare"
	az storage share create -n $(file_share_name) --account-name $(azure_storage_account) --account-key $(azure_storage_key)

create-workspace:
	az batchai workspace create -n $(WORKSPACE) -g $(GROUP_NAME)

create-experiment:
	az batchai experiment create -n $(EXPERIMENT) -g $(GROUP_NAME) -w $(WORKSPACE)

create-cluster: set-storage
	az batchai cluster create \
	-w $(WORKSPACE) \
	--name ${CLUSTER_NAME} \
	--image UbuntuLTS \
	--vm-size ${VM_SIZE} \
	--min ${NUM_NODES} --max ${NUM_NODES} \
	--afs-name ${FILE_SHARE_NAME} \
	--afs-mount-path extfs \
	--user-name mat \
	--password dnstvxrz \
	--storage-account-name $(STORAGE_ACCOUNT_NAME) \
	--storage-account-key $(azure_storage_key)

show-cluster:
	az batchai cluster show -n ${CLUSTER_NAME} -w $(WORKSPACE)

list-clusters:
	az batchai cluster list -w $(WORKSPACE) -o table

list-nodes:
	az batchai cluster list-nodes -n ${CLUSTER_NAME} -w $(WORKSPACE) -o table

test:
	$(call generate_job_spec, --prefetch)
	$(call submit_job, ${JOB_NAME})

list-jobs:
	az batchai job list -w $(WORKSPACE) -e $(EXPERIMENT) -o table

list-files:
	az batchai job file list -w $(WORKSPACE) -e $(EXPERIMENT) --j ${JOB_NAME} --output-directory-id stdouterr

stream-stdout:
	$(call stream_stdout, ${JOB_NAME})

stream-stderr:
	$(call stream_stderr, ${JOB_NAME})

delete-job:
	$(call delete_job, ${JOB_NAME})

delete-experiment:
	az batchai experiment delete -w $(WORKSPACE) --name ${EXPERIMENT} -g ${GROUP_NAME} -y

delete-cluster:
	az configure --defaults group=''
	az configure --defaults location=''
	az batchai cluster delete -w $(WORKSPACE) --name ${CLUSTER_NAME} -g ${GROUP_NAME} -y

delete: delete-cluster
	az batchai experiment delete -w $(WORKSPACE) --name ${experiment} -g ${GROUP_NAME} -y
	az batchai workspace delete -w ${WORKSPACE} -g ${GROUP_NAME} -y
	az group delete --name ${GROUP_NAME} -y


setup: select-subscription set-az-defaults create-experiment


###### Submit Jobs ######

submit-jobs:
	$(call generate_job_spec, --mount --copy_to_local --prefetch)
	$(call submit_job, local_prefetch)
	$(call generate_job_spec, --mount --copy_to_local)
	$(call submit_job, local_noprefetch)
	$(call generate_job_spec, --mount --prefetch)
	$(call submit_job, mount_prefetch)
	$(call generate_job_spec, --mount)
	$(call submit_job, mount_noprefetch)
	$(call generate_job_spec, --prefetch)
	$(call submit_job, az_blob_prefetch)
	$(call generate_job_spec,)
	$(call submit_job, az_blob_no_prefetch)

####### delete jobs ######
clean-jobs:
	$(call delete_job, local_prefetch)
	$(call delete_job, local_noprefetch)
	$(call delete_job, mount_prefetch)
	$(call delete_job, mount_noprefetch)
	$(call delete_job, az_blob_prefetch)
	$(call delete_job, az_blob_no_prefetch)

###### Gather Results ######

gather-results:results.json
	@echo "All results gathered"

results.json: local_prefetch.stdout local_prefetch.stderr mount_prefetch.stdout mount_prefetch.stderr az_blob_prefetch.stdout az_blob_prefetch.stderr local_noprefetch.stdout local_noprefetch.stderr mount_noprefetch.stdout mount_noprefetch.stderr az_blob_noprefetch.stdout az_blob_noprefetch.stderr
	grep RESULT seed*.results > results.txt

local_prefetch.stdout:
	$(call stream_stdout, local_prefetch)>local_prefetch.stdout
local_prefetch.stderr:
	$(call stream_stderr, local_prefetch)>local_prefetch.stderr
local_noprefetch.stdout:
	$(call stream_stdout, local_noprefetch)>local_noprefetch.stdout
local_noprefetch.stderr:
	$(call stream_stderr, local_noprefetch)>local_noprefetch.stderr
mount_prefetch.stdout:
	$(call stream_stdout, mount_prefetch)>mount_prefetch.stdout
mount_prefetch.stderr:
	$(call stream_stderr, mount_prefetch)>mount_prefetch.stderr
mount_noprefetch.stdout:
	$(call stream_stdout, mount_noprefetch)>mount_noprefetch.stdout
mount_noprefetch.stderr:
	$(call stream_stderr, mount_noprefetch)>mount_noprefetch.stderr
az_blob_prefetch.stdout:
	$(call stream_stdout, az_blob_prefetch)>az_blob_prefetch.stdout
az_blob_prefetch.stderr:
	$(call stream_stderr, az_blob_prefetch)>az_blob_prefetch.stderr
az_blob_noprefetch.stdout:
	$(call stream_stdout, az_blob_no_prefetch)>az_blob_noprefetch.stdout
az_blob_noprefetch.stderr:
	$(call stream_stderr, az_blob_no_prefetch)>az_blob_noprefetch.stderr


clean-results:
	rm *.results
	rm results.txt

make plot: results.json
	python ../produce_plot.py

.PHONY: help run-local-test
