from helper_functions import download_model_from_gcs
from helper_functions import download_test_df_from_gcs

download_model_from_gcs(destination_path='/home/model-server/model/best-model.pth')
download_test_df_from_gcs(destination_path='/home/model-server/test_df.csv')