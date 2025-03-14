# Axolotl API Documentation with quartodoc

This directory contains the API documentation for Axolotl, automatically generated using quartodoc.

## Setup

1. Make sure quartodoc is installed:
   ```
   pip install quartodoc
   ```

2. Install Quarto (required to render the documentation):
   ```
   # Download and install the latest Quarto release
   # Visit https://quarto.org/docs/get-started/ for installation instructions
   ```

## Generating Documentation

Run the documentation generation script:
```
python scripts/generate_docs.py
```

This will:
- Read the configuration from `_quarto.yml`
- Extract documentation from the Python source code
- Generate Quarto markdown files (.qmd) in the `docs/api` directory

## Preview the Documentation

After generating the documentation, preview it with:
```
quarto preview
```

## Building the Site

Build the complete site with:
```
quarto render
```

This will create a `_site` directory with the static HTML site.

## Configuration

The documentation generation is configured in two places:

1. `_quarto.yml` - Contains the `quartodoc` section that defines which modules to document
2. The API section in the Quarto website sidebar configuration (also in `_quarto.yml`)

## Customization

To customize the documentation, you can:

1. Add more modules to document in the `quartodoc` section of `_quarto.yml`
2. Create template files in the `quartodoc_templates` directory
3. Adjust the layout in the Quarto configuration
