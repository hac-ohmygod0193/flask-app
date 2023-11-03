from flask import Flask, request, render_template, send_from_directory
from pathlib import Path
from pytube import YouTube
import os
import torch
import whisper
from whisper.utils import get_writer
from werkzeug.utils import secure_filename
from flask import send_file
from zipfile import ZipFile

app = Flask(__name__)

# Define the path for uploading and storing files
UPLOAD_FOLDER = './temp'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # Max file size (16MB)
ALLOWED_EXTENSIONS = set(['mp3'])

# Use CUDA, if available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)

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
def transcribe_file(model, file, plain, srt, vtt, tsv):
    """
    Runs Whisper on an audio file

    Parameters
    ----------
    model: Whisper
        The Whisper model instance.

    file: str
        The file path of the file to be transcribed.

    plain: bool
        Whether to save the transcription as a text file or not.

    srt: bool
        Whether to save the transcription as an SRT file or not.

    vtt: bool
        Whether to save the transcription as a VTT file or not.

    tsv: bool
        Whether to save the transcription as a TSV file or not.

    download: bool
        Whether to download the transcribed file(s) or not.

    Returns
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
    if plain:
        print(f"\nCreating text file")
        # Save as a TXT file with hard line breaks
        txt_writer = get_writer("txt", output_directory)
        txt_writer(result, str(file_path.stem),options)
        
        result_file = "./temp/"+str(file_path.stem)+'.txt'
        files_to_send.append(result_file)
    if srt:
        print(f"\nCreating SRT file")
        srt_writer = get_writer("srt", output_directory)
        srt_writer(result, str(file_path.stem), options)
        
        result_file = "./temp/"+str(file_path.stem)+'.srt'
        files_to_send.append(result_file)
    if vtt:
        print(f"\nCreating VTT file")
        vtt_writer = get_writer("vtt", output_directory)
        vtt_writer(result, str(file_path.stem), options)
        
        result_file = "./temp/"+str(file_path.stem)+'.vtt'
        files_to_send.append(result_file)
    if tsv:
        print(f"\nCreating TSV file")

        tsv_writer = get_writer("tsv", output_directory)
        tsv_writer(result, str(file_path.stem), options)
        
        result_file = "./temp/"+str(file_path.stem)+'.tsv'
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
