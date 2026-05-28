"""Bedrock에서 현재 사용 가능한 Claude 모델 목록 확인"""
import boto3
from dotenv import load_dotenv

load_dotenv()

bedrock = boto3.client(service_name="bedrock", region_name="us-east-1")
res = bedrock.list_foundation_models(byProvider="Anthropic")

for m in res["modelSummaries"]:
    print(m["modelId"])
