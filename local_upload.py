import os
import sys
import boto3
import json
import streamlit as st
import time
import pandas as pd
import cv2  # still used to save images out
import numpy as np
from decord import VideoReader
from decord import cpu, gpu
from video_to_frames_decord import extract_frames, video_to_frames
import base64



def transcribe_file(audio_file_name, transcribe_client):
    #job_name is the name of the path to the file, without the S3:// at the beginning. this will be the unique transcribe job identifier
    job_name = (audio_file_name.split('.')[0]).replace(" ", "")    
    job_name = job_name[5:].replace("/", "_")
    
    file_format = audio_file_name.split('.')[1]
    
    if (check_job_name(job_name, transcribe_client) == False):
        print(f"Job name {job_name} is taken. Time to check if it's completed.")
    else:
        print(f"No job with the name {job_name}, let's start one!")
        transcribe_client.start_transcription_job(
          TranscriptionJobName=job_name,
          Media={'MediaFileUri': audio_file_name},
          MediaFormat = file_format,
          LanguageCode='en-US'
        )
    
    max_tries = 60
    while max_tries > 0:
        max_tries -= 1
        job = transcribe_client.get_transcription_job(TranscriptionJobName=job_name)
        job_status = job["TranscriptionJob"]["TranscriptionJobStatus"]
        if job_status in ["COMPLETED", "FAILED"]:
            print(f"Job {job_name} is {job_status}.")
            if job_status == "COMPLETED":
                data = pd.read_json(job['TranscriptionJob']['Transcript']['TranscriptFileUri'])
                return data['results'][1][0]['transcript']
            break
        else:
            print(f"Waiting for {job_name}. Current status is {job_status}.")
        time.sleep(10)    

# write a function to get a list of current Transcribe jobs, and then to check if an amazon Transcribe job already exists with the full name of the file in S3
def check_job_name(job_name, transcribe_client):
    current_jobs = transcribe_client.list_transcription_jobs()
    for job in current_jobs['TranscriptionJobSummaries']:
        if job_name == job['TranscriptionJobName']:
            return False
    return job_name


        
def get_summmary(transcript):
    brt = boto3.client(service_name='bedrock-runtime')
    prompt = '"\n\nHuman: Summarize in 2-3 lines: "' + transcript + '\n\nAssistant:"'
    body = json.dumps({
        "prompt": prompt,
        "max_tokens_to_sample": 300,
        "temperature": 0.1,
        "top_p": 0.9,
    })
    modelId = 'anthropic.claude-v2'
    accept = 'application/json'
    contentType = 'application/json'
    response = brt.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)
    response_body = json.loads(response.get('body').read())
    # text
    summary = response_body.get('completion')
    print(summary)
    return(summary)


def upload_to_s3(uploaded_file, bucket, s3_path):
    if uploaded_file is not None:
        s3 = boto3.client('s3')
        filename = uploaded_file.name
        s3.upload_fileobj(uploaded_file, bucket, filename)
        #wait until s3.upload_fileobj job has completed and the file has fully uploaded
        while True:
            try:
                s3.head_object(Bucket=bucket, Key=filename)
                st.write(filename, "successfully uploaded to S3")
                break
            except:
                time.sleep(1)
        s3_path = "s3://" + bucket + "/" + filename
    return s3_path

def call_haiku(base64_string, bedrock_client):
    prompt_config = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 4096,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": frame_base64_str,
                        },
                    },
                    {"type": "text", "text": "Provide a caption for this image in one sentence"},
                ],
            }
        ],
    }

    body = json.dumps(prompt_config)

    modelId = "anthropic.claude-3-haiku-20240307-v1:0"
    accept = "application/json"
    contentType = "application/json"

    response = bedrock_client.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    response_body = json.loads(response.get("body").read())

    results = response_body.get("content")[0].get("text")
    return results

'''
st.title('video summarizer') 
st.write('For uploaded videos') 
'''
uploaded_file = False
uploaded_file = st.file_uploader("Choose a file")
s3_path = ""
bucket = "dundundundundundun"

submit_button = st.button("Submit", type="primary")
if submit_button and uploaded_file:
    filename = uploaded_file.name
    video_to_frames(filename, uploaded_file, frames_dir='frames', overwrite=False, every=100)
    # create a bedrock runtime client
    bedrock_client = boto3.client(service_name='bedrock-runtime')
    for frame in os.listdir('frames/animal-short-clip.mp4'):
        if frame == ".ipynb_checkpoints":
            continue
        frame_path = 'frames/animal-short-clip.mp4/' + frame
        # read the frame as a numpy array
        st.write(frame_path)
        frame_array = cv2.imread(frame_path)
        # convert the frame to a base64-encoded string
        frame_base64 = cv2.imencode('.jpg', frame_array)[1].tobytes()
        # create a base64-encoded string from the frame
        frame_base64_str = base64.b64encode(frame_base64).decode('utf-8')
        results = call_haiku(frame_base64_str, bedrock_client)
        st.write(results)
    s3_path = upload_to_s3(uploaded_file, bucket, s3_path)
    transcribe_client = boto3.client("transcribe")
    full_transcript = transcribe_file(s3_path, transcribe_client)
    print("Full transcript:", full_transcript)
    #st.write same as above print statement
    st.write("Full transcript:", full_transcript)
    summary = get_summmary(full_transcript)
    #st.write same as above print statement
    st.write(summary)
