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
import collections


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
    # wait until the Transcribe job finishes
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

        
# write a function to get a list of current Transcribe jobs, and then to check if an amazon Transcribe job already exists for this file
def check_job_name(job_name, transcribe_client):
    current_jobs = transcribe_client.list_transcription_jobs()
    for job in current_jobs['TranscriptionJobSummaries']:
        if job_name == job['TranscriptionJobName']:
            return False
    return job_name


# not used anymore       
def get_transcript_summmary(transcript, bedrock_client):
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


# get the full summary based on both audio and visuals
def get_AV_summmary(transcript, frame_descriptions, bedrock_client):
    prompt = '"\n\nHuman: Your goal is to give me a comprehensive understanding of a video, using a transcript of the audio and descriptions of the frames. Give me a summary of the video, and tell me what the purpose of the video is. This should be no more than 5 sentences. You can include descriptions of individual frames and excerpts from the transcript to make your case. This is the transcript: "' + transcript + "\n and this is the descriptions of the frames, displayed in a dictionary: " + str(frame_descriptions) + '\n\nAssistant:"'
    body = json.dumps({
        "prompt": prompt,
        "max_tokens_to_sample": 300,
        "temperature": 0.1,
        "top_p": 0.9,
    })
    modelId = 'anthropic.claude-v2'
    accept = 'application/json'
    contentType = 'application/json'
    response = bedrock_client.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)
    response_body = json.loads(response.get('body').read())
    # text
    summary = response_body.get('completion')
    print(summary)
    return(summary)


def upload_to_s3(uploaded_file, bucket, s3_path, s3_client):
    if uploaded_file is not None:
        filename = uploaded_file.name
        s3_client.upload_fileobj(uploaded_file, bucket, filename)
        # wait until s3.upload_fileobj job has completed and the file has fully uploaded
        while True:
            try:
                s3_client.head_object(Bucket=bucket, Key=filename)
                st.write(filename, "Successfully uploaded to S3")
                break
            except:
                time.sleep(1)
        s3_path = "s3://" + bucket + "/" + filename
    return s3_path


# this makes individual calls to claude for each extracted frame
def get_frame_descriptions(base64_string, bedrock_client):
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
                    {"type": "text", "text": "Provide a caption for this image in one sentence. Use clear, straightforward language, without embellishing with subjective adjectives."},
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


# initializing variables
s3_path = "" # will define later
bucket = "dundundundundundun" # s3 bucket name to store the video, for transcribe to pull from
frames_dir = "frames" # name of the top level diectory to store frames
uploaded_file = False # will become true when a file is uploaded

# create the AWS clients
bedrock_client = boto3.client(service_name='bedrock-runtime')
transcribe_client = boto3.client("transcribe")
s3_client = boto3.client('s3')

st.title('Video Summarizer') 
st.write('For uploaded videos only, use the other file for youtube URLs') 
uploaded_file = st.file_uploader("Choose a file")
submit_button = st.button("Submit", type="primary")

if submit_button and uploaded_file:
    filename = uploaded_file.name
    
    # we're sampling every 100 frames, hence every = 100
    video_to_frames(filename, uploaded_file, frames_dir, overwrite=False, every=100)
       
    uploaded_file.seek(0)
    s3_path = upload_to_s3(uploaded_file, bucket, s3_path, s3_client)
    full_transcript = transcribe_file(s3_path, transcribe_client)
    
    results_dict ={}
    for frame in os.listdir(frames_dir + "/" + filename):
        if frame == ".ipynb_checkpoints":
            continue
        frame_path = frames_dir + "/" + filename + "/" + frame
        # read the frame as a numpy array
        frame_array = cv2.imread(frame_path)
        # convert the frame to a base64-encoded string
        frame_base64 = cv2.imencode('.jpg', frame_array)[1].tobytes()
        # create a base64-encoded string from the frame
        frame_base64_str = base64.b64encode(frame_base64).decode('utf-8')  
        # call bedrock
        results = get_frame_descriptions(frame_base64_str, bedrock_client)
        results_dict[frame] = results
        
    #get the dictionary in order, so that we're able to see the frame descriptions in the order they appear 
    sorted_results_dict = collections.OrderedDict(sorted(results_dict.items()))
    summary = get_AV_summmary(full_transcript, sorted_results_dict, bedrock_client)

    st.write("These are explanations of some of the frames in the video, thanks to Claude v3 Haiku:\n")
    st.write(sorted_results_dict)
    st.write("Full transcript from Amazon Transcribe:\n", full_transcript)
    st.write("This is an explanation of the video written by Claude v2, given the transcript and the scene descriptions:\n")
    st.write(summary)
