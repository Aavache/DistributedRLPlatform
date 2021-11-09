'''This python script serves to other scripts to connect to AWS services'''
import boto3
import os
import pandas as pd
import utils

class ClusterConnector():
    def __init__(self, opt):
        # Parameters
        self.game_id = opt['env']['env_name']
        self.stream_name = opt['kinesis']['stream_name']
        self.stream_arn = opt['kinesis']['stream_arn']
        self.partition_key = opt['kinesis']['partition_key']
        self.bucket_name = opt['s3']['bucket_name']
        self.weight_file_name =  opt['s3']['weight_name']
        self.weight_local_path = os.path.join(opt['weight_dir'], opt['experiment_name'], self.weight_file_name)
        utils.mkdir(os.path.join(opt['weight_dir'], opt['experiment_name']))

        self.verbose = opt['verbose']

        # Connector to kinesis to upload game records
        self.kinesis_client = boto3.client('kinesis') # This takes the default configuration setted in aws configure, otherwise load from file
        # Check the status of the stream 
        print("Kinesis connection: {}".format(self.kinesis_client.describe_stream_summary(StreamName=self.stream_name)))

        # Connect to S3 bucket for model retreival
        self.s3_client = boto3.client('s3') 

        self.upload_count = 0
        self.download_count = 0

    def upload_batch(self, batch, col_names):
        '''Sending a experience collected by the client'''
        for sample in batch:
            df = pd.DataFrame([sample], columns=col_names)
            df['game_id'] = self.game_id # Partition key in this simplified case
            json_data = df.to_json(orient='records') # String in text format
            data = bytes(json_data, encoding='utf-8')
            records = [{'Data':data, 'PartitionKey':self.partition_key}]
            try:
                self.kinesis_client.put_records(Records=records, StreamName=self.stream_name)
            except Exception as ex:
                print("ERROR: occured while uploading batch {}: {}".format(self.upload_count, ex))
        self.upload_count += 1
        if self.verbose: 
            print('INFO: {}th records with a total of {} samples have been successfully uploaded!'.format(
                                                                    self.upload_count, len(batch)))

    def download_model(self):
        '''Downloading the new updated model trained in the cluster'''
        try:
            self.s3_client.download_file(self.bucket_name, self.weight_file_name, self.weight_local_path)
            self.download_count += 1
            if self.verbose: 
                print('INFO: {}th models successfully downloaded!'.format(self.upload_count))
        except Exception as ex:
            print("ERROR occured while downloading weights {}: {}".format(self.download_count, ex))


        