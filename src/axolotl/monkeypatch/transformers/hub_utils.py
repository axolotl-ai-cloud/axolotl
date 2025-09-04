# Patch transformers list_repo_templates. Remove when
# https://github.com/huggingface/transformers/pull/40669 merged

from pathlib import Path
from typing import Optional

from huggingface_hub import list_repo_tree, snapshot_download
from huggingface_hub.utils import (
    GatedRepoError,
    LocalEntryNotFoundError,
    OfflineModeIsEnabled,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)
from requests.exceptions import HTTPError
import requests


def _list_repo_templates(
    repo_id: str,
    *,
    local_files_only: bool,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> list[str]:
    """List template files from a repo.

    A template is a jinja file located under the `additional_chat_templates/` folder.
    If working in offline mode or if internet is down, the method will list jinja template from the local cache - if any.
    """

    from transformers.utils import hub

    CHAT_TEMPLATE_DIR = getattr(hub, "CHAT_TEMPLATE_DIR", "additional_chat_templates")

    if not local_files_only:
        try:
            return [
                entry.path.removeprefix(f"{CHAT_TEMPLATE_DIR}/")
                for entry in list_repo_tree(
                    repo_id=repo_id,
                    revision=revision,
                    path_in_repo=CHAT_TEMPLATE_DIR,
                    recursive=False,
                )
                if entry.path.endswith(".jinja")
            ]
        except (GatedRepoError, RepositoryNotFoundError, RevisionNotFoundError):
            raise  # valid errors => do not catch
        except (HTTPError, OfflineModeIsEnabled, requests.exceptions.ConnectionError):
            pass  # offline mode, internet down, etc. => try local files

    # check local files
    try:
        snapshot_dir = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=True,
        )
    except LocalEntryNotFoundError:  # No local repo means no local files
        return []
    templates_dir = Path(snapshot_dir, CHAT_TEMPLATE_DIR)
    if not templates_dir.is_dir():
        return []
    return [
        entry.stem
        for entry in templates_dir.iterdir()
        if entry.is_file() and entry.name.endswith(".jinja")
    ]


def apply_list_repo_templates_patch():
    from transformers.utils import hub

    if not hasattr(hub, "list_repo_templates_axolotl"):
        hub.list_repo_templates_axolotl = True
        hub.list_repo_templates = _list_repo_templates
