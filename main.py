from flask import Flask, request, render_template, send_from_directory, redirect, url_for,jsonify
from pathlib import Path
from pytube import YouTube
import os
import torch
import whisper
from whisper.utils import get_writer
from werkzeug.utils import secure_filename
from flask import send_file
from zipfile import ZipFile
from claude_api import Client
import time

app = Flask(__name__)
# Initialize your Claude API client with the session cookie
cookie = "sessionKey=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"
#cookie = "sessionKey=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"
#cookie = "sessionKey=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"

claude_api = Client(cookie)

# Define the path for uploading and storing files
UPLOAD_FOLDER = '/mnt/c/Users/ohmygod/OneDrive - 國立陽明交通大學/side-project/Note-taking-gpt/flask/temp'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # Max file size (16MB)
ALLOWED_EXTENSIONS = set(['pdf','txt', 'png', 'jpg', 'jpeg', 'gif','mp3','mp4'])

# Use CUDA, if available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#print(DEVICE)

# Load the Whisper model and other functions here
model = whisper.load_model("base").to(DEVICE)

def to_snake_case(name):
    return name.lower().replace(" ", "_").replace(":", "_").replace("__", "_")

def download_youtube_audio(url,  file_name = None):
    "Download the audio from a YouTube video"
    target_path = "./temp"

    yt = YouTube(url)

    video = yt.streams.filter(only_audio=True).first()

    out_file = video.download(output_path=target_path)

    video_title, ext = os.path.splitext(out_file)
    file_name = video_title + '.mp3'
    os.rename(out_file, file_name)

    print("target path = " + (file_name))
    print("mp3 has been successfully downloaded.")
    return file_name
def download_multiple_files(files_to_send):
    zip_filename = './temp/transripts.zip'  # Name for the zip file

    with ZipFile(zip_filename, 'w') as zipf:
        for file in files_to_send:
            zipf.write(file, os.path.basename(file))  # Add each file to the zip
            os.remove(file)

    response = send_file(zip_filename, as_attachment=True)
    os.remove(zip_filename)  # Remove the temporary zip file

    return response

def save_response_to_markdown(filename,response,files_to_send):
    # Create a unique filename (e.g., using a timestamp)
    name = filename.split('.')[0]
    response_filename = f"{name}.md"
    # Specify the directory to save the file (e.g., within the UPLOAD_FOLDER)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], response_filename)
    
    # Save the response as a Markdown file
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(response)
    files_to_send.append(file_path)
    return response_filename 
def claude(file,filename,files_to_send):
    print("Sending....",filename)
    summary_prompt = "這份txt檔是一段我自己整理的語音轉文字的逐字稿。我需要詳細內容摘要並以Markdown格式輸出，順便請幫我校正語音轉文字過程中的錯字、標點符號錯誤。"
    class_note_taking_prompt = "這份txt檔是一段我自己整理的語音轉文字的逐字稿。我需要利用康乃爾筆記法（Cornell Note-taking Method)、子彈筆記（Bullet Journal，簡稱BuJo）、心智圖（Mind Map)、圖表筆記法（Charting）四種方法整理一份詳細筆記、以及相關使用到的數學公式，並以Markdown格式輸出，如有使用數學符號請用latex表示,心智圖內容指定使用mermaid語言的代碼塊格式。並請幫我校正語音轉文字過程中的錯字、標點符號錯誤。"
    prompt = summary_prompt
    conversation_id = claude_api.create_new_chat()['uuid']
    summary_response = claude_api.send_message(summary_prompt, conversation_id,attachment=file,timeout=600)
    time.sleep(1)
    class_note_taking_response = claude_api.send_message(class_note_taking_prompt, conversation_id,timeout=600)
    response = summary_response+'\n'+class_note_taking_response
    print(response)

    response_filename = save_response_to_markdown(filename,response,files_to_send)
    deleted = claude_api.delete_conversation(conversation_id)
    if deleted:
        print("Conversation deleted successfully")
    else:
        print("Failed to delete conversation")
    #return redirect(url_for('result',filename=response_filename))

def transcribe_file(model, file, plain, srt, vtt, tsv):
    """
    Runs Whisper on an audio file
    -------
    A dictionary containing the resulting text ("text") and segment-level details ("segments"), and
    the spoken language ("language"), which is detected when `decode_options["language"]` is None.
    """
    files_to_send = []
    file_path = Path(file)
    output_directory = file_path.parent

    # Run Whisper
    result = model.transcribe(file, verbose = False)

    # Set some initial options values
    options = {
        'max_line_width': None,
        'max_line_count': None,
        'highlight_words': False
    }
    print(f"\nCreating text file")
    # Save as a TXT file with hard line breaks
    txt_writer = get_writer("txt", output_directory)
    txt_writer(result, str(file_path.stem),options)
    txt_file = os.path.join(app.config['UPLOAD_FOLDER'], str(file_path.stem)+'.txt')
    claude(txt_file,str(file_path.stem),files_to_send)
    if plain:
        result_file = os.path.join(app.config['UPLOAD_FOLDER'], str(file_path.stem)+'.txt')
        files_to_send.append(result_file)
    if srt:
        print(f"\nCreating SRT file")
        srt_writer = get_writer("srt", output_directory)
        srt_writer(result, str(file_path.stem), options)
        
        result_file = os.path.join(app.config['UPLOAD_FOLDER'], str(file_path.stem)+'.srt')
        files_to_send.append(result_file)
    if vtt:
        print(f"\nCreating VTT file")
        vtt_writer = get_writer("vtt", output_directory)
        vtt_writer(result, str(file_path.stem), options)
        
        result_file = os.path.join(app.config['UPLOAD_FOLDER'], str(file_path.stem)+'.vtt')
        files_to_send.append(result_file)
    if tsv:
        print(f"\nCreating TSV file")

        tsv_writer = get_writer("tsv", output_directory)
        tsv_writer(result, str(file_path.stem), options)
        
        result_file = os.path.join(app.config['UPLOAD_FOLDER'], str(file_path.stem)+'.tsv')
        files_to_send.append(result_file)
    
    # Clean up the temporary file
    os.remove(file)
    response = download_multiple_files(files_to_send)
    return response


@app.route('/')
def index():
    return render_template('index.html')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS
@app.route('/transcribe', methods=['POST'])
def transcribe():
    input_format = request.form.get('input_format')
    plain = request.form.get('plain') == 'true'
    srt = request.form.get('srt') == 'true'
    vtt = request.form.get('vtt') == 'true'
    tsv = request.form.get('tsv') == 'true'
    #download = request.form.get('download') == 'true'

    if input_format == 'youtube':
        # Handle YouTube transcription
        url = request.form.get('url')
        # Implement the YouTube transcription logic here
        # Download the audio stream of the YouTube video
        audio = download_youtube_audio(url)
        print(f"Downloading audio stream: {audio}")
        # Transcribe the audio stream
        result = transcribe_file(model, audio, plain, srt, vtt, tsv)
    elif input_format == 'local':
        # Handle local file transcription
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print(file)
            # Implement the local file transcription logic here
            # You can use 'temp.mp3' as the path to the uploaded file
            # Transcribe the specified local audio file
            result = transcribe_file(model, os.path.join(app.config['UPLOAD_FOLDER'], filename), plain, srt, vtt, tsv)
    return result

if __name__ == '__main__':
    app.run(debug=True)
