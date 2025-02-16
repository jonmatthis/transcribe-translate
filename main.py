import asyncio
import csv
import logging
import os
import uuid
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from moviepy import VideoFileClip
from openai import AsyncOpenAI
from pydantic import BaseModel
from pydub import AudioSegment

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
client = AsyncOpenAI(api_key=openai_api_key)


class TimestampEntry(BaseModel):
    start: float
    end: float
    language: str
    language_code: str


class ProcessedSegment(BaseModel):
    start: float
    end: float
    language: str
    transcript: str
    translation: str


def parse_time(seconds: float) -> str:
    """Convert seconds to SRT timestamp format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace('.', ',')


def parse_timestamps(csv_path: str) -> List[TimestampEntry]:
    """Parse timestamps and language data from a CSV file."""
    logging.info(f"Parsing timestamps from CSV: {csv_path}")
    entries = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            entries.append(TimestampEntry(
                start=float(row['start_time']),
                end=float(row['end_time']),
                language=row['language'].lower(),
                language_code=row['language_code']
            ))
    logging.info(f"Parsed {len(entries)} timestamp entries")
    return entries


async def process_segment(segment: AudioSegment,
                          language_name: str,
                          language_code: str,
                          translation_target_language_name: str,
                          start_time: float) -> List[
    ProcessedSegment]:
    """Process an audio segment with transcription and translation."""
    logging.info(f"Processing segment of language {language_name}")
    chunks = []
    start_times = []
    max_duration = 10.0
    for start in range(0, len(segment), int(max_duration * 1000)):
        end = min(start + int(max_duration * 1000), len(segment))
        chunks.append(segment[start:end])
        start_times.append(start_time + start / 1000)

    results = []
    tasks = []
    for start_time, chunk in zip(start_times, chunks):
        tasks.append(asyncio.create_task(transcribe_and_translate(chunk=chunk,
                                                                  segment_start_time=start_time,
                                                                  spoken_language_code=language_code,
                                                                  spoken_language_name=language_name,
                                                                  translation_target_language_name=translation_target_language_name,
                                                                  results=results)))

    await asyncio.gather(*tasks)
    return results


async def transcribe_and_translate(chunk: AudioSegment,
                                   segment_start_time: float,
                                   spoken_language_code: str,
                                   spoken_language_name: str,
                                   translation_target_language_name: str,
                                   results: List[ProcessedSegment]):
    temp_path = f"temp_segment_{uuid.uuid4()}.mp3"
    chunk.export(temp_path, format="mp3")
    logging.debug(f"Exported chunk to {temp_path}")
    with open(temp_path, "rb") as f:
        transcript = await  client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            language=spoken_language_code
        )
    translation = await client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "system",
            "content": f"Translate this text from {spoken_language_name} ({spoken_language_code}) to  {translation_target_language_name}. Preserve punctuation."
        }, {
            "role": "user",
            "content": transcript.text
        }]
    )
    os.remove(temp_path)
    logging.debug(f"Removed temporary file {temp_path}")
    results.append(ProcessedSegment(
        start=segment_start_time,
        end=segment_start_time + len(chunk) / 1000,
        language=spoken_language_name,
        transcript=transcript.text,
        translation=translation.choices[0].message.content
    ))


def generate_srt(segments: List[ProcessedSegment], target_lang: str, output_path: str):
    """Generate an SRT file for a specific language."""
    logging.info(f"Generating SRT file for {target_lang} at {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        for idx, segment in enumerate(segments, 1):
            f.write(f"{idx}\n")
            f.write(f"{parse_time(segment.start)} --> {parse_time(segment.end)}\n")
            content = segment.transcript if segment.language == target_lang else segment.translation
            f.write(f"{content}\n\n")


def generate_combined_srt(segments: List[ProcessedSegment], output_path: str):
    """Generate a combined bilingual SRT file."""
    logging.info(f"Generating combined bilingual SRT at {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        for idx, segment in enumerate(segments, 1):
            f.write(f"{idx}\n")
            f.write(f"{parse_time(segment.start)} --> {parse_time(segment.end)}\n")

            main_text = f"(spoken - {segment.language}) {segment.transcript}"
            translation_lang = 'english' if segment.language != 'english' else segment.language
            translated_text = f"(translated - {translation_lang}) {segment.translation}"

            f.write(f"{main_text}\n{translated_text}\n\n")


def extract_audio_from_video(video_path: str) -> str:
    """Extract audio from a video file."""
    logging.info(f"Extracting audio from video: {video_path}")
    video = VideoFileClip(video_path)
    audio_path = video_path.rsplit('.', 1)[0] + '.mp3'
    if os.path.exists(audio_path):
        logging.info(f"Audio already exists at {audio_path}")
        return audio_path
    video.audio.write_audiofile(audio_path)
    logging.info(f"Audio extracted to {audio_path}")
    return audio_path


async def main(audio_or_video_path: str,
               translation_target_language_name: str = 'english',
               speaker_csv_path: Optional[str] = None):
    """Main function to process audio or video files and generate subtitles."""
    audio_or_video_path = Path(audio_or_video_path)
    if not speaker_csv_path:
        speaker_csv_path = audio_or_video_path.parent / 'speakers.csv'
    if not audio_or_video_path.exists():
        logging.error(f"Audio or video path not found.")
        raise FileNotFoundError(f"File {audio_or_video_path} not found.")

    output_dir = audio_or_video_path.parent
    if audio_or_video_path.suffix.lower() in ['.mp4', '.mov', '.avi', '.mkv']:
        logging.info("Detected video file. Extracting audio...")
        audio_path = extract_audio_from_video(str(audio_or_video_path))
    else:
        audio_path = audio_or_video_path

    logging.info(f"Loading audio from {audio_path}")
    audio = AudioSegment.from_file(audio_path)
    timestamps = parse_timestamps(speaker_csv_path)

    logging.info("Beginning processing of audio segments")
    processed_segments = []
    for timestamp in timestamps:
        chunk = audio[int(timestamp.start * 1000):int(timestamp.end * 1000)]
        segments = await process_segment(segment=chunk,
                                         language_name=timestamp.language,
                                         language_code=timestamp.language_code,
                                         translation_target_language_name=translation_target_language_name,
                                         start_time=timestamp.start)
        processed_segments.extend(segments)

    processed_segments.sort(key=lambda x: x.start)

    generate_srt(processed_segments, 'english', os.path.join(output_dir, 'subtitles_english.srt'))
    generate_srt(processed_segments, 'turkish', os.path.join(output_dir, 'subtitles_turkish.srt'))
    generate_combined_srt(processed_segments, os.path.join(output_dir, 'subtitles_bilingual.srt'))
    logging.info("Subtitle generation complete")


if __name__ == "__main__":
    outer_audio_or_video_path = "turkish-vid/turkish-vid.mp4"
    asyncio.run(main(audio_or_video_path=outer_audio_or_video_path))
