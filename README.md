# Distributed RL Training Platform

This repository contains the Python-AWS implementation of the platform. The platform is composed of two stages, the ingestion and the training stages. For the first stage, we choose a pipeline by combining Kinesis streams, Apache Spark cluster and Cassandra database. Finally the second stage carries out the training with EC2 instances with ray-RLLIB training cluster as well as a S3 bucket to store the updated models for the public access. Take a look to the following figure for better understanding.

![Alt text](figures/distributed_diag.png)

### 1. Implementation

As the code has to be executed in different machines, we separate the implementation in different folders.  

### 2. Instalation.

In case you implementation is on AWS services, you may have to create an account and set-up certain things such as IAM roles and credentials for accessing from your local machine.

#### 2.1. Kinesis Streams

Type Kinesis on the search box and you hit create a new streams. You will be able to set certain parameters such as the shard number and the retention time. Keep the stream name and the ARN location of the stream since the clients need it for uploading.

#### 2.2. Apache Spark on EC2.  

First, EC2 instances have to be initialized(we use a ubuntu medium instance). Each of the instances do as follows:

- Configure the security rules for inbound and outbound traffic.
- Access the instance via terminal.
- Install java and python.
- Install pip and the cassandra-driver library.
- Upload your pyspark script to the instance and run it via <code>spark-submit</code> command with the appropiate kinesis and cassandra packages.
- You may also be interested to access the spark dashboard by accessing the port 4040.

#### 2.3. Cassandra on EC2

- Configure the security rules for inbound and outbound traffic.
- Access the instance via terminal.
- Install java.
- Install Cassandra in each node.
- Set up the Cassandra configuration file to operate with your setup (assigning the addresses for each node).
- You may access the Cassandra cql terminal by typing <code>cqlsh</code> in the command line.

#### 2.4. Ray cluster on EC2  

- Configure the security rules for inbound and outbound traffic.
- Access the instance via terminal.
- Install python 3 and pip
- Install the following dependencies: pytorch, ray, ray[default], ray[tune], ray[rllib]
- Run the code in your master node and connect your worker nodes to the master node using the correct ip address.

#### 2.5. S3 bucket

A bucket can be easily created on AWS services. Simply type S3 on the search box and carry out the necessary steps.

#### Client

Each client should have installed python with the following dependencies: pytorch, numpy, pandas, boto3, gym,  matplotlib(optional). Once it is downloaded, run the script experience_collection.py.