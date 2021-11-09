'''This is a pyspark script for  kinesis-Cassandra integration '''
from pyspark import SparkContext, SparkConf
from pyspark.sql import Row
from pyspark.streaming import StreamingContext
from pyspark.streaming.kinesis import KinesisUtils, InitialPositionInStream
import cassandra
from cassandra.cluster import Cluster
from datetime import datetime as dt

# Running the code for a single node
# ./spark-2.4.8-bin-hadoop2.7/bin/spark-submit --master local[3] --packages com.datastax.spark:spark-cassandra-connector_2.11:2.4.1,org.apache.spark:spark-streaming-kinesis-asl_2.11:2.4.7 kinesis_to_cassandra.py 

class ConnectionPool(object):
	'''Singleton static connection pool for cassandra database'''
	__connection = None
	@staticmethod
	def get_connection(ip, port, keyspace):
		if ConnectionPool.__connection is None:
			ConnectionPool(ip, port, keyspace)
		return ConnectionPool.__connection

	def __init__(self, ip, port, keyspace):
		# print('Creating new connection')
		cassandra_conn = Cluster(contact_points=(ip,), port=port)
		self.session = cassandra_conn.connect(keyspace)
		ConnectionPool.__connection = self

def parse(line):
	'''parsing streaming json records'''
	# Removing unnecessary symbols
	bad_symbols = ['(',')','{','}','"', '[', ']']
	for symbol in bad_symbols:
	    line = line.replace(symbol, '')

	# Splitting key-values in lists
	line = line.split(':')
	s = []
	for substring in line:
	    s+=substring.split(',')

	# Converting to Row format
	Record = Row(s[0],s[2],s[4],s[9],s[11],s[13],s[18],s[20])
	return Record(int(s[1]), int(s[3]), [float(s[5]),float(s[6]),float(s[7]),float(s[8])],
				int(s[10]), float(s[12]), [float(s[14]),float(s[15]),float(s[16]),float(s[17])],
				'true'==s[19].lower(),s[21])

def InsertToCassandra(rdd):
	'''
	Sending records to Cassandra.
		- Reference: https://spark.apache.org/docs/latest/streaming-programming-guide.html
	'''
	# cassandra_conn = Cluster(contact_points=("3.36.111.128",), port=9042)
	# session = cassandra_conn.connect('test')
	conn = ConnectionPool.get_connection("3.36.133.139", 9042, 'test')
	try:
		conn.session.execute("""INSERT INTO cartpole_sac (pk, episode, step, state, action, reward, next_state, done, game_id) 
								VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)""", 
								(cassandra.util.uuid_from_time(dt.now()), rdd[0],rdd[1],rdd[2],rdd[3],rdd[4],rdd[5],rdd[6],rdd[7]))
	except Exception as e:
		ConnectionPool.__connection = None
		print('ERROR: Failed to insert to cassandra. {}'.format(str(e)))
	# session.shutdown()

if __name__ == "__main__":
	# Kinesis params
	kinesis_app_name = "game_records"
	kinesis_stream_name = "distributed_game_rl_stream"
	end_point_url = "kinesis.ap-northeast-2.amazonaws.com"
	region= "ap-northeast-2"
	init_pos = InitialPositionInStream.TRIM_HORIZON
	checkpoint_interval = 10

	# Cassandra params
	# cassandra_address = "52.78.96.201" # global?
	# cassandra_port = 9042
		
	sc = SparkContext(appName="KinesisToCassandra")
	sc.setLogLevel("ERROR")
	ssc = StreamingContext(sc, 1)

	# Creating kinesis stream
	lines = KinesisUtils.createStream(ssc, 	
									kinesis_app_name, 
									kinesis_stream_name,
									end_point_url,region,
									init_pos,
									checkpoint_interval)

	row = lines.map(lambda x: parse(x)) # Formating input streams 
	row.pprint() # Printing for debugging purposes

	# row.foreachRDD(lambda rdd: rdd.foreach(InsertToCassandra))
	row.foreachRDD(lambda rdd: rdd.foreach(InsertToCassandra)) # Inserting records in cassandra

	ssc.start()
	ssc.awaitTermination()  # Wait for the computation to terminate