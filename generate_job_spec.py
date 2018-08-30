import argparse
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_job_dict(image_name, prefetch=True, mount=True, copy_to_local=True, dataset='imagenet2012'):
    cmd = ''
    if copy_to_local:
        cmd = 'cp -r $AZ_BATCHAI_JOB_MOUNT_ROOT/data /data && '
    cmd += 'bash /research/resnet/official/resnet/run_az_blob.sh sDGa69CWFFjo7yswO1AH9OZKukfhjFOagz0l8ed7LfLi0AZTOEZrXc/GQCw/5sdCSFxVxosmJqB3e0KGAMnZsA== %s ' % dataset
    if copy_to_local:
        cmd += '-dd /data '
    elif mount:
        cmd += '-dd $AZ_BATCHAI_JOB_MOUNT_ROOT/data '
    else:
        cmd += '-dd az://mlperfstorage.blob.core.windows.net/%s ' % dataset
    if prefetch:
        cmd += '--prefetch '
    else:
        cmd += '--noprefetch '
    job_spec = {
        "$schema": "https://raw.githubusercontent.com/Azure/BatchAI/master/schemas/2017-09-01-preview/job.json",
        "properties": {
            "containerSettings": {
                "imageSourceRegistry": {
                    "image": image_name,
                    "serverUrl": "mlperfregistry.azurecr.io",
                    "credentials": {
                        "username": "mlperfregistry",
                        "password": "CamocqUfj41tIotScXKr4/pY7NGirpBV"
                    }
                }
            },
            "customToolkitSettings": {
                "commandLine": cmd
            },
            "nodeCount": 1,
            "stdOutErrPathPrefix": "$AZ_BATCHAI_JOB_MOUNT_ROOT/logs",
            "mountVolumes": {
                "azureFileShares": [
                    {
                        "azureFileUrl": "https://mlperfstorage.file.core.windows.net/logs",
                        "relativeMountPath": "logs"
                    }
                ]
            }
        }
    }
    if mount or copy_to_local:
        job_spec['properties']['mountVolumes']['azureBlobFileSystems'] = \
                [{
                    'accountName': 'mlperfstorage',
                    'containerName': dataset,
                    'relativeMountPath': 'data'
                }]
    return job_spec


def write_json_to_file(json_dict, filename, mode='w'):
    with open(filename, mode) as outfile:
        json.dump(json_dict, outfile, indent=4, sort_keys=True)
        outfile.write('\n\n')


def main(image_name,
         filename='job.json',
         prefetch=True,
         mount=True,
         copy_to_local=True,
         dataset='imagenet2012'):
    logger.info('Creating manifest {} with {} image...'.format(filename, image_name))
    job_template = generate_job_dict(image_name,
                                     prefetch=prefetch,
                                     mount=mount,
                                     copy_to_local=copy_to_local,
                                     dataset=dataset)
    write_json_to_file(job_template, filename)
    logger.info('Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate manifest')
    parser.add_argument('--docker_image', type=str,
                        help='docker image to use')
    parser.add_argument('--filename', '-f', dest='filename', type=str, nargs='?',
                        default='job.json',
                        help='name of the file to save job spec to')
    parser.add_argument('--prefetch', action='store_true')
    parser.add_argument('--mount', action='store_true')
    parser.add_argument('--copy_to_local', action='store_true')
    parser.add_argument('--dataset', type=str, default='imagenet2012',
                        help='dataset used to be trained resnet.')
    args = parser.parse_args()
    main(args.docker_image,
         filename=args.filename,
         prefetch=args.prefetch,
         mount=args.mount,
         copy_to_local=args.copy_to_local,
         dataset=args.dataset)
