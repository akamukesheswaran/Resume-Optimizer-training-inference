in order to run the inference notebook, aka the "demo" notebook, 
- you need to download the saved_models folder located in the "inference_notebook_rltd" folder and then you need to zip it and then paste it into the files in the inference notebook. keep it zipped. there is a cell that will unzip it. Do not worry about that.
- in the inference notebook, you need to add a folder called "uploads".
- in there, you need to upload a resume and a job description. it has to be .txt, .pdf, .docx
- the resume has to say "resume" in it
- the job description has to say "job_description" in it
-
- cell 7 shows you the contents of the job description and resume
- the last cell shows you the skill extraction and the matching

- training notebook:
- name: CSE6363-Training.ipynb
- how to run:
- once you open it in google colab, do not run it yet. You need to add these files from this github first:
- 1. UpdatedResumeDataset.csv
  2. you need to add a kaggle token. download from kaggle. call it: kaggle_token.json
  3. vocab.txt  from: inference_notebook_rltd/saved_models/data/vocab.txt

- with these files, you can now run the training notebook.
- The "validation" notebook is the same as the cse6363-Training.ipynb, it just has the validation cells at the end, too.
