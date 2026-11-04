"""The CI Modal image, defined once so the single- and multi-GPU apps share one build"""

import os
import pathlib
import tempfile

import jinja2
import modal
from jinja2 import select_autoescape

CICD_PATH = pathlib.Path(__file__).parent.resolve()


def ci_image(df_args: dict[str, str]) -> modal.Image:
    template_env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(searchpath=CICD_PATH),
        autoescape=select_autoescape(),
    )
    template = template_env.get_template(
        os.environ.get("E2E_DOCKERFILE", "Dockerfile-uv.jinja")
    )
    # deps_hash.py pins BASE_IMAGE to a digest; the tag is the fallback outside CI
    base_image = (
        os.environ.get("BASE_IMAGE")
        or f"axolotlai/axolotl-base-uv:{df_args['BASE_TAG']}"
    )
    dockerfile = pathlib.Path(tempfile.mkdtemp()) / "Dockerfile"
    dockerfile.write_text(
        template.render(**df_args, BASE_IMAGE=base_image), encoding="utf-8"
    )
    # Builder 2025.06+ (MODAL_IMAGE_BUILDER_VERSION in the workflows) adds no client-dependency
    # layer; 2024.10 pip-installs into the image's python, which the pip-less uv venv lacks.
    return (
        modal.Image.from_dockerfile(dockerfile)
        # cwd leads sys.path; from the image's checkout it shadows the cicd/ Modal mounts at /root
        .workdir("/root")
        .env(df_args)
    )
