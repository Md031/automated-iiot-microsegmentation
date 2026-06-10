from dataclasses import dataclass
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, LongType
import random

# PCA threshold
variance_threshold=0.995

# seed
SEED = 10

# Datasets path
iotid20_path = "Dataset/iotId20"
unsw_path = "Dataset/UNSW_NB15"
iot23_path = "Dataset/iot23_small"

dt_max_depth = 25

@dataclass
class DatasetConfig:
	name: str
	path: str
	saving_path: str
	pca_path: str
	optics_path: str
	dt_path: str
	schema: StructType
	uid: str
	noisy_features: list[str] # features that only add noise to the dataset and should be dropped
	label_features: list[str] # label columns that should not be normalized on the code
	pca_expl_var: float
	sample_size: int
	network_features: dict[str, str]


schema_iot23 = StructType([
		StructField("ts", DoubleType(), True),
		StructField("uid", StringType(), True),
		StructField("id_orig_h", StringType(), True),
		StructField("id_orig_p", IntegerType(), True),
		StructField("id_resp_h", StringType(), True),
		StructField("id_resp_p", IntegerType(), True),
		StructField("proto", StringType(), True),
		StructField("service", StringType(), True),
		StructField("duration", DoubleType(), True),
		StructField("orig_bytes", LongType(), True),
		StructField("resp_bytes", LongType(), True),
		StructField("conn_state", StringType(), True),
		StructField("local_orig", IntegerType(), True),
		StructField("local_resp", IntegerType(), True),
		StructField("missed_bytes", IntegerType(), True),
		StructField("history", StringType(), True),
		StructField("orig_pkts", IntegerType(), True),
		StructField("orig_ip_bytes", LongType(), True),
		StructField("resp_pkts", IntegerType(), True),
		StructField("resp_ip_bytes", LongType(), True),
		StructField("tunnel_parents", StringType(), True),
		StructField("label", StringType(), True),
		StructField("detailed_label", StringType(), True)
])

schema_unsw = StructType([
	StructField("srcip", StringType(), True),
	StructField("sport", IntegerType(), True),
	StructField("dstip", StringType(), True),
	StructField("dsport", IntegerType(), True),
	StructField("proto", StringType(), True),
	StructField("state", StringType(), True),
	StructField("dur", DoubleType(), True),
	StructField("sbytes", IntegerType(), True),
	StructField("dbytes", IntegerType(), True),
	StructField("sttl", IntegerType(), True),
	StructField("dttl", IntegerType(), True),
	StructField("sloss", IntegerType(), True),
	StructField("dloss", IntegerType(), True),
	StructField("service", StringType(), True),
	StructField("Sload", DoubleType(), True),
	StructField("Dload", DoubleType(), True),
	StructField("Spkts", IntegerType(), True),
	StructField("Dpkts", IntegerType(), True),
	StructField("swin", IntegerType(), True),
	StructField("dwin", IntegerType(), True),
	StructField("stcpb", IntegerType(), True),
	StructField("dtcpb", IntegerType(), True),
	StructField("smeansz", IntegerType(), True),
	StructField("dmeansz", IntegerType(), True),
	StructField("trans_depth", IntegerType(), True),
	StructField("res_bdy_len", IntegerType(), True),
	StructField("Sjit", DoubleType(), True),
	StructField("Djit", DoubleType(), True),
	StructField("Stime", LongType(), True),
	StructField("Ltime", LongType(), True),
	StructField("Sintpkt", DoubleType(), True),
	StructField("Dintpkt", DoubleType(), True),
	StructField("tcprtt", DoubleType(), True),
	StructField("synack", DoubleType(), True),
	StructField("ackdat", DoubleType(), True),
	StructField("is_sm_ips_ports", IntegerType(), True),
	StructField("ct_state_ttl", IntegerType(), True),
	StructField("ct_flw_http_mthd", IntegerType(), True),
	StructField("is_ftp_login", IntegerType(), True),
	StructField("ct_ftp_cmd", IntegerType(), True),
	StructField("ct_srv_src", IntegerType(), True),
	StructField("ct_srv_dst", IntegerType(), True),
	StructField("ct_dst_ltm", IntegerType(), True),
	StructField("ct_src_ltm", IntegerType(), True),
	StructField("ct_src_dport_ltm", IntegerType(), True),
	StructField("ct_dst_sport_ltm", IntegerType(), True),
	StructField("ct_dst_src_ltm", IntegerType(), True),
	StructField("attack_cat", StringType(), True),
	StructField("Label", IntegerType(), True)
])

schema_iotid20 = StructType([
	StructField("Flow_ID", StringType(), True),
	StructField("Src_IP", StringType(), True),
	StructField("Src_Port", IntegerType(), True),
	StructField("Dst_IP", StringType(), True),
	StructField("Dst_Port", IntegerType(), True),
	StructField("Protocol", IntegerType(), True),
	StructField("Timestamp", StringType(), True),
	StructField("Flow_Duration", IntegerType(), True),
	StructField("Tot_Fwd_Pkts", IntegerType(), True),
	StructField("Tot_Bwd_Pkts", IntegerType(), True),
	StructField("TotLen_Fwd_Pkts", DoubleType(), True),
	StructField("TotLen_Bwd_Pkts", DoubleType(), True),
	StructField("Fwd_Pkt_Len_Max", DoubleType(), True),
	StructField("Fwd_Pkt_Len_Min", DoubleType(), True),
	StructField("Fwd_Pkt_Len_Mean", DoubleType(), True),
	StructField("Fwd_Pkt_Len_Std", DoubleType(), True),
	StructField("Bwd_Pkt_Len_Max", DoubleType(), True),
	StructField("Bwd_Pkt_Len_Min", DoubleType(), True),
	StructField("Bwd_Pkt_Len_Mean", DoubleType(), True),
	StructField("Bwd_Pkt_Len_Std", DoubleType(), True),
	StructField("Flow_Byts/s", DoubleType(), True),
	StructField("Flow_Pkts/s", DoubleType(), True),
	StructField("Flow_IAT_Mean", DoubleType(), True),
	StructField("Flow_IAT_Std", DoubleType(), True),
	StructField("Flow_IAT_Max", DoubleType(), True),
	StructField("Flow_IAT_Min", DoubleType(), True),
	StructField("Fwd_IAT_Tot", DoubleType(), True),
	StructField("Fwd_IAT_Mean", DoubleType(), True),
	StructField("Fwd_IAT_Std", DoubleType(), True),
	StructField("Fwd_IAT_Max", DoubleType(), True),
	StructField("Fwd_IAT_Min", DoubleType(), True),
	StructField("Bwd_IAT_Tot", DoubleType(), True),
	StructField("Bwd_IAT_Mean", DoubleType(), True),
	StructField("Bwd_IAT_Std", DoubleType(), True),
	StructField("Bwd_IAT_Max", DoubleType(), True),
	StructField("Bwd_IAT_Min", DoubleType(), True),
	StructField("Fwd_PSH_Flags", IntegerType(), True),
	StructField("Bwd_PSH_Flags", IntegerType(), True),
	StructField("Fwd_URG_Flags", IntegerType(), True),
	StructField("Bwd_URG_Flags", IntegerType(), True),
	StructField("Fwd_Header_Len", IntegerType(), True),
	StructField("Bwd_Header_Len", IntegerType(), True),
	StructField("Fwd_Pkts/s", DoubleType(), True),
	StructField("Bwd_Pkts/s", DoubleType(), True),
	StructField("Pkt_Len_Min", DoubleType(), True),
	StructField("Pkt_Len_Max", DoubleType(), True),
	StructField("Pkt_Len_Mean", DoubleType(), True),
	StructField("Pkt_Len_Std", DoubleType(), True),
	StructField("Pkt_Len_Var", DoubleType(), True),
	StructField("FIN_Flag_Cnt", IntegerType(), True),
	StructField("SYN_Flag_Cnt", IntegerType(), True),
	StructField("RST_Flag_Cnt", IntegerType(), True),
	StructField("PSH_Flag_Cnt", IntegerType(), True),
	StructField("ACK_Flag_Cnt", IntegerType(), True),
	StructField("URG_Flag_Cnt", IntegerType(), True),
	StructField("CWE_Flag_Count", IntegerType(), True),
	StructField("ECE_Flag_Cnt", IntegerType(), True),
	StructField("Down/Up_Ratio", DoubleType(), True),
	StructField("Pkt_Size_Avg", DoubleType(), True),
	StructField("Fwd_Seg_Size_Avg", DoubleType(), True),
	StructField("Bwd_Seg_Size_Avg", DoubleType(), True),
	StructField("Fwd_Byts/b_Avg", IntegerType(), True),
	StructField("Fwd_Pkts/b_Avg", IntegerType(), True),
	StructField("Fwd_Blk_Rate_Avg", IntegerType(), True),
	StructField("Bwd_Byts/b_Avg", IntegerType(), True),
	StructField("Bwd_Pkts/b_Avg", IntegerType(), True),
	StructField("Bwd_Blk_Rate_Avg", IntegerType(), True),
	StructField("Subflow_Fwd_Pkts", IntegerType(), True),
	StructField("Subflow_Fwd_Byts", IntegerType(), True),
	StructField("Subflow_Bwd_Pkts", IntegerType(), True),
	StructField("Subflow_Bwd_Byts", IntegerType(), True),
	StructField("Init_Fwd_Win_Byts", IntegerType(), True),
	StructField("Init_Bwd_Win_Byts", IntegerType(), True),
	StructField("Fwd_Act_Data_Pkts", IntegerType(), True),
	StructField("Fwd_Seg_Size_Min", IntegerType(), True),
	StructField("Active_Mean", DoubleType(), True),
	StructField("Active_Std", DoubleType(), True),
	StructField("Active_Max", DoubleType(), True),
	StructField("Active_Min", DoubleType(), True),
	StructField("Idle_Mean", DoubleType(), True),
	StructField("Idle_Std", DoubleType(), True),
	StructField("Idle_Max", DoubleType(), True),
	StructField("Idle_Min", DoubleType(), True),
	StructField("Label", StringType(), True),
	StructField("Cat", StringType(), True),
	StructField("Sub_Cat", StringType(), True)
])

DATASET_CONFIG ={
	"iot23": DatasetConfig(
		name = "iot23",
		path = iot23_path,
		saving_path = f"{iot23_path}/preprocessed_data",
		pca_path = f"{iot23_path}/pca_data",
		optics_path = f"{iot23_path}/optics_clustered",
		dt_path = f"{iot23_path}/dt_model",
		schema = schema_iot23,
		uid = "uid",
		noisy_features = ["uid", "id_orig_h", "id_orig_p", "id_resp_h", "id_resp_p", "local_orig", "local_resp", "ts"],
		label_features = ["label", "detailed_label"],
		pca_expl_var = 0.95,
		sample_size = 1000,
		network_features = {
			"srcIp": "id_orig_h",
			"srcPort": "id_orig_p",
			"dstIp": "id_resp_h",
			"dstPort": "id_resp_p",
			"protocol": "proto"
		}
	),
	"iotid20": DatasetConfig(
		name = "iotid20",
		path = iotid20_path,
		saving_path = f"{iotid20_path}/preprocessed_data",
		pca_path = f"{iotid20_path}/pca_data",
		optics_path = f"{iotid20_path}/optics_clustered",
		dt_path = f"{iotid20_path}/dt_model",
		schema = schema_iotid20,
		uid = "uid",
		noisy_features = ["Flow_ID", "Src_IP", "Src_Port", "Dst_IP", "Dst_Port"],
		label_features = ["Label", "Cat", "Sub_Cat"],
		pca_expl_var = 20,
		sample_size = 1000,
		network_features = {
			"srcIp": "Src_IP",
			"srcPort": "Src_Port",
			"dstIp": "Dst_IP",
			"dstPort": "Dst_Port",
			"protocol": "Protocol"
		}
	),
	"unsw": DatasetConfig(
		name = "unsw",
		path = unsw_path,
		saving_path = f"{unsw_path}/preprocessed_data",
		pca_path = f"{unsw_path}/pca_data",
		optics_path = f"{unsw_path}/optics_clustered",
		dt_path = f"{unsw_path}/dt_model",
		uid = "uid", # no uid is provided so we'll add it manually during loading
		schema = schema_unsw,
		noisy_features = ["uid", "srcip", "sport", "dstip", "dsport"],
		label_features = ["Label", "attack_cat", "is_ftp_login"],
		pca_expl_var = 30,
		sample_size = 1000,
		network_features = {
			"srcIp": "srcip",
			"srcPort": "sport",
			"dstIp": "dstip",
			"dstPort": "dsport",
			"protocol": "proto"
		}
	)
}