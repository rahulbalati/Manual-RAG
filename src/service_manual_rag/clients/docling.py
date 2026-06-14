"""Shared Docling configuration for PDF parsing."""

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption


def create_pdf_converter(
    *,
    generate_page_images: bool = True,
    images_scale: float = 1.0,
) -> DocumentConverter:
    pipeline_options = PdfPipelineOptions(
        generate_page_images=generate_page_images,
        images_scale=images_scale,
    )
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
            ),
        }
    )
