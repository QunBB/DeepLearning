import os
from typing import Dict, Optional, Union, List
from pathlib import Path
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
from huggingface_hub import HfApi, hf_hub_url, cached_download
import fnmatch


def download(model_name_or_path: Optional[str] = None,
             cache_folder: Optional[str] = None,
             ignore_files: Optional[Union[List, str]] = None):
    if model_name_or_path is not None and model_name_or_path != "":
        print("Load pretrained SentenceTransformer: {}".format(model_name_or_path))

        # Old models that don't belong to any organization
        basic_transformer_models = ['albert-base-v1', 'albert-base-v2', 'albert-large-v1', 'albert-large-v2', 'albert-xlarge-v1', 'albert-xlarge-v2', 'albert-xxlarge-v1', 'albert-xxlarge-v2', 'bert-base-cased-finetuned-mrpc', 'bert-base-cased', 'bert-base-chinese', 'bert-base-german-cased', 'bert-base-german-dbmdz-cased', 'bert-base-german-dbmdz-uncased', 'bert-base-multilingual-cased', 'bert-base-multilingual-uncased', 'bert-base-uncased', 'bert-large-cased-whole-word-masking-finetuned-squad', 'bert-large-cased-whole-word-masking', 'bert-large-cased', 'bert-large-uncased-whole-word-masking-finetuned-squad', 'bert-large-uncased-whole-word-masking', 'bert-large-uncased', 'camembert-base', 'ctrl', 'distilbert-base-cased-distilled-squad', 'distilbert-base-cased', 'distilbert-base-german-cased', 'distilbert-base-multilingual-cased', 'distilbert-base-uncased-distilled-squad', 'distilbert-base-uncased-finetuned-sst-2-english', 'distilbert-base-uncased', 'distilgpt2', 'distilroberta-base', 'gpt2-large', 'gpt2-medium', 'gpt2-xl', 'gpt2', 'openai-gpt', 'roberta-base-openai-detector', 'roberta-base', 'roberta-large-mnli', 'roberta-large-openai-detector', 'roberta-large', 't5-11b', 't5-3b', 't5-base', 't5-large', 't5-small', 'transfo-xl-wt103', 'xlm-clm-ende-1024', 'xlm-clm-enfr-1024', 'xlm-mlm-100-1280', 'xlm-mlm-17-1280', 'xlm-mlm-en-2048', 'xlm-mlm-ende-1024', 'xlm-mlm-enfr-1024', 'xlm-mlm-enro-1024', 'xlm-mlm-tlm-xnli15-1024', 'xlm-mlm-xnli15-1024', 'xlm-roberta-base', 'xlm-roberta-large-finetuned-conll02-dutch', 'xlm-roberta-large-finetuned-conll02-spanish', 'xlm-roberta-large-finetuned-conll03-english', 'xlm-roberta-large-finetuned-conll03-german', 'xlm-roberta-large', 'xlnet-base-cased', 'xlnet-large-cased']

        if os.path.exists(model_name_or_path):
            # Load from path
            model_path = model_name_or_path
        else:
            # Not a path, load from hub
            if '\\' in model_name_or_path or model_name_or_path.count('/') > 1:
                raise ValueError("Path {} not found".format(model_name_or_path))

            if '/' not in model_name_or_path and model_name_or_path.lower() not in basic_transformer_models:
                # A model from sentence-transformers
                model_name_or_path = "sentence-transformers/" + model_name_or_path

            model_path = os.path.join(cache_folder, model_name_or_path.replace("/", "_"))

            if os.path.exists(model_path) and os.listdir(model_path):
                print(f'{model_path} exists, will not download model config files, exit!')
                return

            # Download from hub with caching
            snapshot_download(model_name_or_path,
                              cache_dir=cache_folder,
                              library_name='sentence-transformers',
                              ignore_files=['flax_model.msgpack', 'rust_model.ot', 'tf_model.h5', 'model.safetensors'
                                            ] + (ignore_files if isinstance(ignore_files, list) else [ignore_files]))
        return model_path


def snapshot_download(
        repo_id: str,
        revision: Optional[str] = None,
        cache_dir: Union[str, Path, None] = None,
        library_name: Optional[str] = None,
        library_version: Optional[str] = None,
        user_agent: Union[Dict, str, None] = None,
        ignore_files: Optional[List[str]] = None
) -> str:
    """
    Method derived from huggingface_hub.
    Adds a new parameters 'ignore_files', which allows to ignore certain files / file-patterns
    """
    if cache_dir is None:
        cache_dir = HUGGINGFACE_HUB_CACHE
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    _api = HfApi()
    model_info = _api.model_info(repo_id=repo_id, revision=revision)

    storage_folder = os.path.join(
        cache_dir, repo_id.replace("/", "_")
    )

    for model_file in model_info.siblings:
        if ignore_files is not None:
            skip_download = False
            for pattern in ignore_files:
                if fnmatch.fnmatch(model_file.rfilename, pattern):
                    skip_download = True
                    break

            if skip_download:
                continue

        url = hf_hub_url(
            repo_id, filename=model_file.rfilename, revision=model_info.sha
        )
        relative_filepath = os.path.join(*model_file.rfilename.split("/"))

        # Create potential nested dir
        nested_dirname = os.path.dirname(
            os.path.join(storage_folder, relative_filepath)
        )
        os.makedirs(nested_dirname, exist_ok=True)

        path = cached_download(
            url,
            cache_dir=storage_folder,
            force_filename=relative_filepath,
            library_name=library_name,
            library_version=library_version,
            user_agent=user_agent,
        )

        if os.path.exists(path + ".lock"):
            os.remove(path + ".lock")

    return storage_folder
