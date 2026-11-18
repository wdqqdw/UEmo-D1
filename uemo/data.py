"""UEmo-Data-v1 records and the shared multimodal decision prompt."""
import json
from pathlib import Path

LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
SYSTEM = 'You are a helpful assistant. Classify the target using the supplied media and text.'


def validate_record(row):
    for key in ('id', 'source', 'source_id', 'instruction', 'gt_answer'):
        if not isinstance(row.get(key), str) or not row[key].strip():
            raise ValueError(f'Missing/non-string {key}')
    choices = row.get('choices')
    if (not isinstance(choices, list) or not 2 <= len(choices) <= 26
            or any(not isinstance(x, str) or not x for x in choices)
            or len(set(choices)) != len(choices) or row['gt_answer'] not in choices):
        raise ValueError(f'Invalid choices/answer for {row["id"]}')
    if 'subtask' in row and (not isinstance(row['subtask'], str) or not row['subtask']):
        raise ValueError('subtask must be a nonempty string')
    for group in [row] + ([row['context']] if 'context' in row else []):
        if not isinstance(group, dict) or not isinstance(group.get('text'), str):
            raise ValueError('Each target/context needs a text string')
        present = {'text'} if group['text'] else set()
        for kind in ('audio', 'video', 'image'):
            paths = group.get(kind + '_path')
            if not isinstance(paths, list) or any(not isinstance(p, str) or not p for p in paths):
                raise ValueError(f'{kind}_path must be a list of paths')
            if paths:
                present.add(kind)
        modalities = group.get('modality')
        if (not isinstance(modalities, list) or len(modalities) != len(set(modalities))
                or set(modalities) != present):
            raise ValueError('modality must describe the actual target/context fields')
        if group['video_path'] and len(group['video_path']) != len(group['audio_path']):
            raise ValueError('Each video needs its aligned audio at the same list position')


def load_records(path, *, check_media=False, media_root=None):
    path = Path(path)
    with path.open(encoding='utf-8') as handle:
        rows = json.load(handle)
    if not isinstance(rows, list) or not rows:
        raise ValueError('Metadata must be a nonempty JSON list')
    seen = set()
    root = Path(media_root) if media_root else path.parent
    for row in rows:
        validate_record(row)
        if row['id'] in seen:
            raise ValueError(f'Duplicate ID: {row["id"]}')
        seen.add(row['id'])
        if check_media:
            for group in [row] + ([row['context']] if 'context' in row else []):
                for kind in ('audio', 'video', 'image'):
                    for rel in group[kind + '_path']:
                        resolve_media(root, rel)
    return rows


def resolve_media(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root):
        raise ValueError(f'Media path escapes its split directory: {relative}')
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def prepare(processor, row, root):
    # Delayed imports keep schema checks and --help usable without CUDA libraries.
    from PIL import Image
    from .media import read_audio, read_video
    validate_record(row)
    audios, images, videos, content = [], [], [], []
    groups = [('Preceding context:', row['context'])] if 'context' in row else []
    groups.append(('Target to classify:', row))
    for heading, group in groups:
        content.append({'type': 'text', 'text': heading})
        if group['text']:
            content.append({'type': 'text', 'text': group['text']})
        for rel in group['image_path']:
            with Image.open(resolve_media(root, rel)) as image:
                images.append(image.convert('RGB'))
            content.append({'type': 'image', 'image': 'provided image'})
        if group['video_path']:
            for vp, ap in zip(group['video_path'], group['audio_path']):
                videos.append(read_video(resolve_media(root, vp)))
                audios.append(read_audio(resolve_media(root, ap)))
                content.append({'type': 'video', 'video': 'provided audiovisual clip'})
        else:
            for rel in group['audio_path']:
                audios.append(read_audio(resolve_media(root, rel)))
                content.append({'type': 'audio', 'audio': 'provided audio'})
    prompt = row['instruction'] + '\nChoose your answer from:\n'
    prompt += '\n'.join(f'{LETTERS[i]}. {v}' for i, v in enumerate(row['choices']))
    prompt += '\nAnswer with only the single option letter, without explanation.'
    content.append({'type': 'text', 'text': prompt})
    conversation = [
        {'role': 'system', 'content': [{'type': 'text', 'text': SYSTEM}]},
        {'role': 'user', 'content': content},
    ]
    return {'text': processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False),
            'audio': audios, 'images': images, 'videos': videos}


def batch_prepare(processor, prepared):
    aa = [x for p in prepared for x in p['audio']]
    ii = [x for p in prepared for x in p['images']]
    vv = [x for p in prepared for x in p['videos']]
    return processor(
        text=[p['text'] for p in prepared], audio=aa or None,
        images=ii or None, videos=vv or None, return_tensors='pt', padding=True,
        use_audio_in_video=True,
        images_kwargs={'min_pixels': 56*56, 'max_pixels': 448*448},
        videos_kwargs={'fps': 2.0, 'do_sample_frames': False, 'min_pixels': 56*56, 'max_pixels': 224*224},
        audio_kwargs={'max_length': max(map(len, aa)) if aa else 16000, 'truncation': False},
    )


def setting_key(row):
    # CREMA-D audio-only and audiovisual share a subtask name in the release.
    return (row['source'], row.get('subtask', row['source']), tuple(sorted(row['modality'])))
