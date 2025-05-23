import logging
import numpy as np

from anno_page.core.metadata import GraphicalObjectMetadata, RelatedLinesMetadata
from anno_page.enums.language import Language
from anno_page.enums.line_relation import LineRelation


class DummyImageCaptioning:
    def __init__(self, config, device, config_path):
        self.device = device

        self.categories = config["categories"] if "categories" in config else None
        np.random.seed(42)

        self.logger = logging.getLogger(__name__)

    def process_page(self, page_image, page_layout):
        text_lines = list(page_layout.lines_iterator(["text", None]))

        if self.categories is not None:
            for region in page_layout.regions:
                if region.category in self.categories:
                    caption_lines_metadata = None
                    reference_lines_metadata = None

                    if len(text_lines) > 0:
                        np.random.shuffle(text_lines)
                        caption_lines = text_lines[:np.random.randint(2, 5)]
                        self.logger.info(f"Caption lines for {region.id}: {len(caption_lines)}")

                        np.random.shuffle(text_lines)
                        reference_lines = text_lines[:np.random.randint(3, 8)]
                        print(f"Reference lines for {region.id}: {len(reference_lines)}")

                        reference_lines_text = " ".join([line.transcription for line in reference_lines if line.transcription])
                        caption_lines_text = " ".join([line.transcription for line in caption_lines if line.transcription])

                        reference_lines_metadata = RelatedLinesMetadata(tag_id=f"rtf.{region.id}",
                                                                        mods_id=f"MODS_{region.id}_RELATED_0001",
                                                                        lines=reference_lines,
                                                                        relation=LineRelation.REFERENCE,
                                                                        description=reference_lines_text,
                                                                        title=reference_lines_text)

                        caption_lines_metadata = RelatedLinesMetadata(tag_id=f"fc.{region.id}",
                                                                      mods_id=f"MODS_{region.id}_CAPTION_0001",
                                                                      lines=caption_lines,
                                                                      relation=LineRelation.CAPTION,
                                                                      description=caption_lines_text,
                                                                      title=caption_lines_text)

                        for reference_line in reference_lines:
                            reference_line.metadata = [reference_lines_metadata]

                        for caption_line in caption_lines:
                            caption_line.metadata = [caption_lines_metadata]

                    region.metadata = GraphicalObjectMetadata(tag_id=region.id,
                                                              mods_id=f"MODS_{region.id}",
                                                              caption={
                                                                  Language.ENGLISH: "This is a caption",
                                                                  Language.CZECH: "Toto je popis"
                                                              },
                                                              topics={
                                                                  Language.ENGLISH: "Here, are, the, topics",
                                                                  Language.CZECH: "Tady, jsou, témata"
                                                              },
                                                              color={
                                                                  Language.ENGLISH: "color",
                                                                  Language.CZECH: "barva"
                                                              },
                                                              description="Object description",
                                                              title="Object title",
                                                              reference_lines_metadata=reference_lines_metadata,
                                                              caption_lines_metadata=caption_lines_metadata)

        return page_layout
