first run in the sagemaker terminal:
pip install --no-cache-dir -r requirements.txt
OR
sh setup.sh

to run the app, go to terminal and type:
sh run.sh local_upload.py
- this is a shellscript that creates a proxy to allow you to host a streamlit app from sagemaker. if you try to just use the streamlit command it won't work, the two IPs that are generated will not be accessible

summary_from_yt.py works, but it doesn't have the newer functionality and it requires the stupid GCP API key

video_to_frames_decord is from github, i didn't write it. and basically it just pulls frames out of a file, sampling how you want, and stores them in a directory (in our case, frames)

run sh cleanup.sh when you're done, to shut down whatever streamlit stuff is running

much of this from: https://aws.amazon.com/blogs/machine-learning/build-streamlit-apps-in-amazon-sagemaker-studio/

