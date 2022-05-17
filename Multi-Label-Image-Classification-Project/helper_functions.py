from google.cloud import storage

# Initialise a client
storage_client = storage.Client(project='xxxxxxxxxx') # Your GCP Project ID

# Download images from Google Cloud Storage Bucket to docker container
def download_images_from_gcs():
    bucket_name = 'movie-classifier-images-bucket'
    prefix = ''
    dl_dir = './images/'

    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)  # Get list of files
    for blob in blobs:
        filename = blob.name.replace('/', '_') 
        blob.download_to_filename(dl_dir + filename)  # Download

# Download Model from Google Cloud Storage Bucket to docker container
def download_model_from_gcs(destination_path):
    
    bucket_name = 'movie-classifier-model-bucket'
    
    # Create a bucket object for our bucket
    bucket = storage_client.get_bucket(bucket_name)
    # Create a blob object from the filepath
    blob = bucket.blob("model/best-model.pth")
    # Download the file to a destination
    blob.download_to_filename(destination_path)

# Download test_df from Google Cloud Storage Bucket to docker container
def download_test_df_from_gcs(destination_path):
    
    bucket_name = 'multi-label-image-classification-bucket'
    
    # Create a bucket object for our bucket
    bucket = storage_client.get_bucket(bucket_name)
    # Create a blob object from the filepath
    blob = bucket.blob("input/test_df.csv")
    # Download the file to a destination
    blob.download_to_filename(destination_path)
    
# Check if a flie exists on Google Cloud Storage Bucket
def is_file_exists_on_google_cloud_storage(file_name, bucket_name):  
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    stats = storage.Blob(bucket=bucket, name=file_name).exists(storage_client)
    return stats
        
# Delete previous best model from Google Cloud Storage Bucket
def delete_model_from_gcs():
    bucket_name = 'movie-classifier-model-bucket'
    directory_name = 'model/best-model.pth'

    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    # list all objects in the directory
    blobs = bucket.list_blobs(prefix=directory_name)
    for blob in blobs:
        blob.delete()