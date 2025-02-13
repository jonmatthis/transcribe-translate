import csv
import os
import logging
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pydub import AudioSegment
from moviepy import VideoFileClip

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)


def parse_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace('.', ',')


def parse_timestamps(csv_path: str) -> list[dict]:
    logging.info(f"Parsing timestamps from CSV: {csv_path}")
    entries = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            entries.append({
                'start': float(row['start_time']),
                'end': float(row['end_time']),
                'lang': 'en' if row['language'].lower() == 'english' else 'es'
            })
    logging.info(f"Parsed {len(entries)} timestamp entries")
    return entries


def process_segment(segment: AudioSegment, lang: str, max_duration: float = 10.0) -> list[dict]:
    logging.info(f"Processing segment of language {lang}")

    # Split the segment into smaller chunks if necessary
    chunks = []
    for start in range(0, len(segment), int(max_duration * 1000)):
        end = min(start + int(max_duration * 1000), len(segment))
        chunks.append(segment[start:end])

    results = []
    for chunk in chunks:
        temp_path = "temp_segment.mp3"
        chunk.export(temp_path, format="mp3")
        logging.debug(f"Exported chunk to {temp_path}")

        with open(temp_path, "rb") as f:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                language=lang
            ).text

        target_lang = 'es' if lang == 'en' else 'en'
        translation = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "system",
                "content": f"Translate to {target_lang.upper()}. Preserve punctuation."
            }, {
                "role": "user",
                "content": transcript
            }]
        ).choices[0].message.content

        os.remove(temp_path)
        logging.debug(f"Removed temporary file {temp_path}")
        results.append({
            'transcript': transcript,
            'translation': translation,
            'duration': len(chunk) / 1000
        })

    return results


def generate_srt(entries: list[dict], target_lang: str, output_path: str):
    logging.info(f"Generating SRT file for {target_lang} at {output_path}")
    with open(output_path, 'w') as f:
        for idx, entry in enumerate(entries, 1):
            f.write(f"{idx}\n")
            f.write(f"{parse_time(entry['start'])} --> {parse_time(entry['end'])}\n")
            f.write(f"{entry[target_lang]}\n\n")


def generate_combined_srt(entries: list[dict], output_path: str):
    logging.info(f"Generating combined bilingual SRT at {output_path}")
    with open(output_path, 'w') as f:
        for idx, entry in enumerate(entries, 1):
            f.write(f"{idx}\n")
            f.write(f"{parse_time(entry['start'])} --> {parse_time(entry['end'])}\n")

            # Determine original and translated text
            if entry['lang'] == 'en':
                main_text = f"(spoken - en) {entry['en']}"
                translated_text = f"(translated - es) {entry['es']}"
            else:
                main_text = f"(spoken - es) {entry['es']}"
                translated_text = f"(translated - en) {entry['en']} "

            f.write(f"{main_text}\n{translated_text}\n\n")


def extract_audio_from_video(video_path: str) -> str:
    logging.info(f"Extracting audio from video: {video_path}")
    video = VideoFileClip(video_path)
    audio_path = video_path.rsplit('.', 1)[0] + '.mp3'
    if os.path.exists(audio_path):
        logging.info(f"Audio already exists at {audio_path}")
        return audio_path
    video.audio.write_audiofile(audio_path)
    logging.info(f"Audio extracted to {audio_path}")
    return audio_path


def main(audio_or_video_path: str, speaker_csv_path: str):
    audio_or_video_path = Path(audio_or_video_path)
    if not audio_or_video_path.exists():
        logging.error(f"File {audio_or_video_path} not found.")
        raise FileNotFoundError(f"File {audio_or_video_path} not found.")

    output_dir = audio_or_video_path.parent
    # Check if the input file is a video
    if audio_or_video_path.suffix.lower() in ['.mp4', '.mov', '.avi', '.mkv']:
        logging.info("Detected video file. Extracting audio...")
        audio_path = extract_audio_from_video(str(audio_or_video_path))
    else:
        audio_path = audio_or_video_path

    # Load and split audio
    logging.info(f"Loading audio from {audio_path}")
    audio = AudioSegment.from_file(audio_path)
    segments = parse_timestamps(speaker_csv_path)

    # Process all segments
    logging.info("Beginning processing of audio segments")
    srt_entries = []
    for seg in segments:
        chunk = audio[seg['start'] * 1000:seg['end'] * 1000]
        processed_segments = process_segment(chunk, seg['lang'])

        current_time = seg['start']
        for segment in processed_segments:
            srt_entries.append({
                'start': current_time,
                'end': current_time + segment['duration'],
                'lang': seg['lang'],  # Track original language
                'en': segment['transcript'] if seg['lang'] == 'en' else segment['translation'],
                'es': segment['transcript'] if seg['lang'] == 'es' else segment['translation']
            })
            current_time += segment['duration']

    # Sort entries chronologically
    srt_entries.sort(key=lambda x: x['start'])

    # Generate SRT files
    generate_srt(srt_entries, 'en', os.path.join(output_dir, 'subtitles_en.srt'))
    generate_srt(srt_entries, 'es', os.path.join(output_dir, 'subtitles_es.srt'))
    generate_combined_srt(srt_entries, os.path.join(output_dir, 'subtitles_bilingual.srt'))
    logging.info("Subtitle generation complete")

if __name__ == "__main__":
    outer_audio_or_video_path = "/Users/jon/Downloads/mcl_labor_standout_video/mls_labor_union_Standout.mp4"
    outer_speaker_timestamps_csv = "/Users/jon/Downloads/mcl_labor_standout_video/speakers.csv"
    main(audio_or_video_path=outer_audio_or_video_path,
         speaker_csv_path=outer_speaker_timestamps_csv)