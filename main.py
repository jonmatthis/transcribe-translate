import csv
import os

from dotenv import load_dotenv
from openai import OpenAI
from pydub import AudioSegment

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)


def parse_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace('.', ',')


def parse_timestamps(csv_path: str) -> list[dict]:
    entries = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            entries.append({
                'start': float(row['start_time']),
                'end': float(row['end_time']),
                'lang': 'en' if row['language'].lower() == 'english' else 'es'
            })
    return entries


def process_segment(segment: AudioSegment,
                    lang: str) -> tuple[str, str]:
    temp_path = "temp_segment.mp3"
    segment.export(temp_path, format="mp3")

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
    return transcript, translation


def generate_srt(entries: list[dict], target_lang: str, output_path: str):
    with open(output_path, 'w') as f:
        for idx, entry in enumerate(entries, 1):
            f.write(f"{idx}\n")
            f.write(f"{parse_time(entry['start'])} --> {parse_time(entry['end'])}\n")
            f.write(f"{entry[target_lang]}\n\n")


def generate_combined_srt(entries: list[dict], output_path: str):
    """Generate SRT with both languages, tagging translations"""
    with open(output_path, 'w') as f:
        for idx, entry in enumerate(entries, 1):
            f.write(f"{idx}\n")
            f.write(f"{parse_time(entry['start'])} --> {parse_time(entry['end'])}\n")

            # Determine original and translated text
            if entry['lang'] == 'en':
                main_text = entry['en']
                translated_text = f"{entry['es']} (translated)"
            else:
                main_text = entry['es']
                translated_text = f"{entry['en']} (translated)"

            f.write(f"{main_text}\n{translated_text}\n\n")


def main(audio_path: str, csv_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    # Load and split audio
    audio = AudioSegment.from_file(audio_path)
    segments = parse_timestamps(csv_path)

    # Process all segments
    srt_entries = []
    for seg in segments:
        chunk = audio[seg['start'] * 1000:seg['end'] * 1000]
        original, translation = process_segment(chunk, seg['lang'])

        srt_entries.append({
            'start': seg['start'],
            'end': seg['end'],
            'lang': seg['lang'],  # Track original language
            'en': original if seg['lang'] == 'en' else translation,
            'es': original if seg['lang'] == 'es' else translation
        })

    # Sort entries chronologically
    srt_entries.sort(key=lambda x: x['start'])

    # Generate SRT files
    generate_srt(srt_entries, 'en', os.path.join(output_dir, 'subtitles_en.srt'))
    generate_srt(srt_entries, 'es', os.path.join(output_dir, 'subtitles_es.srt'))
    generate_combined_srt(srt_entries, os.path.join(output_dir, 'subtitles_bilingual.srt'))


if __name__ == "__main__":
    outer_audio_path = "sample-audio.m4a"
    outer_save_path = "."
    outer_speaker_timestamps_csv = "speakers.csv"
    main(audio_path=outer_audio_path, csv_path=outer_speaker_timestamps_csv, output_dir=outer_save_path)
