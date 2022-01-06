import boto3, os

# https://betterprogramming.pub/cron-job-patterns-in-aws-126fbf54a276
def lambda_handler(event, context):
    ec2 = boto3.client('ec2')
    ec2.request_spot_instances(
        SpotPrice='<max price for spot instance, ex. 0.1>',
        InstanceCount=1,
        Type='one-time',
        LaunchSpecification={
            'ImageId': '<ami id>',
            'KeyName': '<keypair to use>',
            'InstanceType': '<instance type, ex. "t2.micro">',
            'UserData': '<base64 encoded script goes here>',
            'IamInstanceProfile': {
                'Arn': '<instance role arn>' #make sure this role has terminate EC2 instance access
            },
            'SecurityGroupIds': ['<list of security group ids>']
        }
 )

def s3_download_file(local_filename, bucket_name, remote_filename):
    print("DOWNLOAD FILE FROM S3: s3://" + bucket_name + '/' + remote_filename + ' -> ' + local_filename)
    s3_cl = boto3.client('s3', endpoint_url=os.getenv('BOTO3_S3_ENDPOINT_URL'))
    with open(local_filename, 'wb') as f:
        s3_cl.download_fileobj(bucket_name, remote_filename, f)


def s3_upload_file(local_filename, bucket_name, remote_filename):
    print("UPLOAD FILE TO S3: " + local_filename + ' -> ' + 's3://' + bucket_name + '/' + remote_filename)
    s3_cl = boto3.client('s3', endpoint_url=os.getenv('BOTO3_S3_ENDPOINT_URL'))
    with open(local_filename, 'rb') as f:
        s3_cl.upload_fileobj(f, bucket_name, remote_filename)


def s3_list(bucket_name, prefix_path):
    print("LIST FROM S3: s3://" + bucket_name + "/" + prefix_path)
    s3_cl = boto3.client('s3', endpoint_url=os.getenv('BOTO3_S3_ENDPOINT_URL'))
    keys = []
    # paginated shite
    paginator = s3_cl.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket_name,Prefix=prefix_path)
    for page in pages:
        for obj in page['Contents']:
            keys.append({'name': obj['Key'], 'size': obj['Size']})
    return keys


def s3_download_directory(local_directory, bucket_name, remote_directory):
    print("DOWNLOAD DIRECTORY FROM S3: s3://" + bucket_name + '/' + remote_directory + ' -> ' + local_directory)
    remote_files = s3_list(bucket_name, remote_directory)
    prefix = remote_directory + '/'
    for rf in remote_files:
        remote_file = rf['name']
        size = rf['size']
        if size == 0:
            continue
        stripped_path = remote_file[len(prefix):]
        local_file = os.path.join(local_directory, stripped_path)
        local_file_dir = os.path.dirname(local_file)
        os.makedirs(local_file_dir, exist_ok=True)
        s3_download_file(local_file, bucket_name, remote_file)

def hack_reader(f = r'C:\Users\david\Dropbox\mobilier\crypto\archived data\ftx\20211017_30590_all.txt\datacollection\ftx/20211017_30590_all.txt'):
    i = 0
    with open(f, 'r') as fp:
        while True:
            i += 1
            if i == 100:
                break
            t = fp.readline()
            print(t)

if False:
    directory='C:/Users/david/Dropbox/mobilier/crypto/archived data/ftx/'
    for filename in os.listdir(directory):
        if (".parquet" in filename):
            s3_upload_file(directory+filename,'gof.crypto.shared', 'ftx_'+filename)

