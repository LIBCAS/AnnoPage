import uuid
from typing import Optional, Dict, List
from lxml import etree as ET
from datetime import datetime

from pero_ocr.core.layout import TextLine

from anno_page.enums.language import Language
from anno_page.enums.category import Category
from anno_page.enums.line_relation import LineRelation


class BaseMetadata:
    def __init__(self, tag_id, mods_id, mods_uuid=None):
        self.tag_id = tag_id
        self.mods_id = mods_id
        self.mods_uuid = uuid.uuid4() if mods_uuid is None else mods_uuid

    @staticmethod
    def _add_genre_element(mods, mods_namespace, language, content, genre_type=None):
        genre = ET.SubElement(mods, f"{{{mods_namespace}}}genre")
        genre.attrib["altRepGroup"] = "genre-1"

        if language is not None:
            genre.attrib["lang"] = language

        if genre_type is not None:
            genre.attrib["type"] = genre_type

        genre.text = content

    @staticmethod
    def _add_type_of_resource_element(mods, mods_namespace, resource_type):
        type_of_resource = ET.SubElement(mods, f"{{{mods_namespace}}}typeOfResource")
        type_of_resource.text = resource_type

    @staticmethod
    def _add_related_item_element(mods, mods_namespace, item_type, item_id):
        related_item = ET.SubElement(mods, f"{{{mods_namespace}}}relatedItem")
        related_item.attrib["type"] = item_type
        related_item.attrib["IDREF"] = item_id

    @staticmethod
    def _add_title_element(mods, mods_namespace, title, language=None):
        title_info = ET.SubElement(mods, f"{{{mods_namespace}}}titleInfo")
        title_info.attrib["altRepGroup"] = "title-1"

        if isinstance(title, dict):
            for lang, text in title.items():
                title_element = ET.SubElement(title_info, f"{{{mods_namespace}}}title")
                title_element.text = text
                title_element.attrib["lang"] = str(lang)

        else:
            title_element = ET.SubElement(title_info, f"{{{mods_namespace}}}title")
            title_element.text = title

            if language is not None:
                title_element.attrib["lang"] = language

    @staticmethod
    def _add_identifier_element(mods, mods_namespace, identifier_value, identifier_type="uuid"):
        identifier = ET.SubElement(mods, f"{{{mods_namespace}}}identifier")
        identifier.text = f"uuid:{identifier_value}"
        identifier.attrib["type"] = identifier_type

    @staticmethod
    def _add_record_info_element(mods, mods_namespace, confidence=None):
        record_info = ET.SubElement(mods, f"{{{mods_namespace}}}recordInfo")

        record_creation_date = ET.SubElement(record_info, f"{{{mods_namespace}}}recordCreationDate")
        record_creation_date.text = datetime.now().isoformat(timespec='seconds')

        record_content_source = ET.SubElement(record_info, f"{{{mods_namespace}}}recordContentSource")
        record_content_source.text = "Orbis Pictus"

        description_standard = ET.SubElement(record_info, f"{{{mods_namespace}}}descriptionStandard")
        description_standard.text = "StandardNDK"

        record_origin = ET.SubElement(record_info, f"{{{mods_namespace}}}recordOrigin")
        record_origin.text = "machine-generated"

        if confidence is not None:
            record_info_note = ET.SubElement(record_info, f"{{{mods_namespace}}}recordInfoNote")
            record_info_note.attrib["type"] = "confidence"
            record_info_note.text = f"{confidence:.3f}"

        record_identifier = ET.SubElement(record_info, f"{{{mods_namespace}}}recordIdentifier")
        record_identifier.attrib["source"] = "Orbis Pictus"
        record_identifier.text = f"uuid:{uuid.uuid4()}"

        language_of_cataloging = ET.SubElement(record_info, f"{{{mods_namespace}}}languageOfCataloging")

        language_term_en = ET.SubElement(language_of_cataloging, f"{{{mods_namespace}}}languageTerm")
        language_term_en.text = str(Language.ENGLISH)
        language_term_en.attrib["authority"] = "iso639-2b"

        language_term_cz = ET.SubElement(language_of_cataloging, f"{{{mods_namespace}}}languageTerm")
        language_term_cz.text = str(Language.CZECH)
        language_term_cz.attrib["authority"] = "iso639-2b"

    def to_altoxml(self, *args, **kwargs):
        raise NotImplementedError


class RelatedLinesMetadata(BaseMetadata):
    def __init__(self,
                 tag_id,
                 mods_id,
                 lines: List[TextLine],
                 relation: LineRelation,
                 description: Optional[str] = None,
                 title: Optional[str | Dict[Language, str]] = None,
                 mods_uuid=None):
        super().__init__(tag_id, mods_id, mods_uuid)
        self.lines = lines
        self.relation = relation
        self.description = description
        self.title = title

    def to_altoxml(self, tags, mods_namespace, confidence, related_mods_id=None):
        if self.relation == LineRelation.REFERENCE:
            values = {
                "tag": "OtherTag",
                "type": "Content",
                "label": "RelatedToFigure",
                "genre": "relatedToFigure",
                "genre_en": "related to figure",
                "genre_cz": "související s obrázkem",
                "related_item_type": "references"
            }
        elif self.relation == LineRelation.CAPTION:
            values = {
                "tag": "StructureTag",
                "type": "Functional",
                "label": "FigureCaption",
                "genre": "caption",
                "genre_en": "caption",
                "genre_cz": "popis",
                "related_item_type": "host"
            }
        else:
            raise ValueError(f"Unknown relation type: {self.relation}")

        tag = ET.SubElement(tags, values["tag"])
        xml_data = ET.SubElement(tag, "XmlData")
        mods = ET.SubElement(xml_data, f"{{{mods_namespace}}}mods")

        tag.set("ID", self.tag_id)
        tag.set("TYPE", values["type"])
        tag.set("LABEL", values["label"])

        if self.description is not None:
            tag.set("DESCRIPTION", self.description)

        mods.set("ID", self.mods_id)

        if self.title is not None:
            self._add_title_element(mods, mods_namespace, title=self.title)

        self._add_type_of_resource_element(mods, mods_namespace, resource_type="text")

        self._add_genre_element(mods, mods_namespace, language=None, genre_type="part", content=values["genre"])
        self._add_genre_element(mods, mods_namespace, language=str(Language.ENGLISH), genre_type="part", content=values["genre_en"])
        self._add_genre_element(mods, mods_namespace, language=str(Language.CZECH), genre_type="part", content=values["genre_cz"])

        self._add_identifier_element(mods, mods_namespace, self.mods_uuid)

        if related_mods_id is not None:
            self._add_related_item_element(mods, mods_namespace, values["related_item_type"], related_mods_id)

        self._add_record_info_element(mods, mods_namespace, confidence=confidence)


class GraphicalObjectMetadata(BaseMetadata):
    def __init__(self, 
                 tag_id,
                 mods_id,
                 mods_uuid=None,
                 caption: Optional[str | Dict[Language, str]] = None,
                 topics: Optional[str | Dict[Language, str]] = None,
                 color: Optional[str | Dict[Language, str]] = None,
                 description: Optional[str] = None,
                 title: Optional[str | Dict[Language, str]] = None,
                 caption_lines_metadata: Optional[RelatedLinesMetadata] = None,
                 reference_lines_metadata: Optional[RelatedLinesMetadata] = None):
        super().__init__(tag_id, mods_id, mods_uuid)
        self.caption = caption
        self.topics = topics
        self.color = color
        self.description = description
        self.title = title

        self.caption_lines_metadata = caption_lines_metadata
        self.reference_lines_metadata = reference_lines_metadata

    def to_altoxml(self, tags, mods_namespace, category, bounding_box, confidence):
        self.graphics_to_altoxml(tags, mods_namespace, category, bounding_box, confidence)

        if self.caption_lines_metadata is not None:
            self.caption_lines_metadata.to_altoxml(tags, mods_namespace, confidence, self.mods_id)

        if self.reference_lines_metadata is not None:
            self.reference_lines_metadata.to_altoxml(tags, mods_namespace, confidence, self.mods_id)

    def graphics_to_altoxml(self, tags, mods_namespace, category, bounding_box, confidence):
        layout_tag = ET.SubElement(tags, "LayoutTag")
        xml_data = ET.SubElement(layout_tag, "XmlData")
        mods = ET.SubElement(xml_data, f"{{{mods_namespace}}}mods")

        mods.set("ID", self.mods_id)

        category = Category.from_string(category)
        category_en = category.to_string(Language.ENGLISH)
        category_mods_en = category.to_string(Language.MODS_GENRE_EN)
        category_mods_cz = category.to_string(Language.MODS_GENRE_CZ)

        layout_tag.set("ID", self.tag_id)
        layout_tag.set("TYPE", "Structural")
        layout_tag.set("LABEL", category_en)

        if self.description is not None:
            layout_tag.set("DESCRIPTION", self.description)

        if self.title is not None:
            self._add_title_element(mods, mods_namespace, title=self.title)

        self._add_type_of_resource_element(mods, mods_namespace, category.to_type_of_resource())

        self._add_genre_element(mods, mods_namespace, str(Language.ENGLISH), category_mods_en)
        self._add_genre_element(mods, mods_namespace, str(Language.CZECH), category_mods_cz)

        self._add_size_element(mods, mods_namespace, bounding_box)

        if self.color is not None:
            self._add_color_elements(mods, mods_namespace)

        if self.caption is not None:
            self._add_caption_elements(mods, mods_namespace)

        if self.topics is not None:
            self._add_topic_elements(mods, mods_namespace)

        self._add_identifier_element(mods, mods_namespace, self.mods_uuid, identifier_type="uuid")

        if self.caption_lines_metadata is not None:
            self._add_related_item_element(mods, mods_namespace, "constituent", self.caption_lines_metadata.mods_id)

        if self.reference_lines_metadata is not None:
            self._add_related_item_element(mods, mods_namespace, "references", self.reference_lines_metadata.mods_id)

        self._add_record_info_element(mods, mods_namespace, confidence)

    @staticmethod
    def _add_size_element(mods, mods_namespace, bounding_box):
        x_min, y_min, x_max, y_max = bounding_box
        width = round(x_max - x_min)
        height = round(y_max - y_min)

        physical_description = ET.SubElement(mods, f"{{{mods_namespace}}}physicalDescription")
        extent = ET.SubElement(physical_description, f"{{{mods_namespace}}}extent")
        extent.text = f"{width}x{height}"
        extent.attrib["unit"] = "pixels"

    def _add_color_elements(self, mods, mods_namespace):
        if isinstance(self.color, str):
            self._add_color_element(mods, mods_namespace, "", self.color)
        elif isinstance(self.color, dict):
            for language, color in self.color.items():
                self._add_color_element(mods, mods_namespace, str(language), color)
        else:
            print(f"Color is not a string or dictionary in GraphicalObjectMetadata '{self.tag_id}'.")

    @staticmethod
    def _add_color_element(mods, mods_namespace, language, color):
        physical_description = ET.SubElement(mods, f"{{{mods_namespace}}}physicalDescription")
        physical_description.attrib["altRepGroup"] = "color-1"
        form = ET.SubElement(physical_description, f"{{{mods_namespace}}}form")
        form.attrib["type"] = "color"
        form.attrib["lang"] = language
        form.text = color

    def _add_caption_elements(self, mods, mods_namespace):
        if isinstance(self.caption, str):
            self._add_caption_element(mods, mods_namespace, "", self.caption)
        elif isinstance(self.caption, dict):
            for language, caption in self.caption.items():
                self._add_caption_element(mods, mods_namespace, str(language), caption)
        else:
            print(f"Caption is not a string or dictionary in GraphicalObjectMetadata '{self.tag_id}'.")

    @staticmethod
    def _add_caption_element(mods, mods_namespace, language, text):
        abstract = ET.SubElement(mods, f"{{{mods_namespace}}}abstract")
        abstract.text = text
        abstract.attrib["altRepGroup"] = "abstract-1"
        abstract.attrib["lang"] = language

    def _add_topic_elements(self, mods, mods_namespace):
        if isinstance(self.topics, str):
            self._add_topic_element(mods, mods_namespace, "", self.topics)
        elif isinstance(self.topics, dict):
            for language, topic in self.topics.items():
                self._add_topic_element(mods, mods_namespace, str(language), topic)
        else:
            print(f"Topic is not a string or dictionary in GraphicalObjectMetadata '{self.tag_id}'.")

    @staticmethod
    def _add_topic_element(mods, mods_namespace, language, text):
        subject = ET.SubElement(mods, f"{{{mods_namespace}}}subject")
        subject.attrib["altRepGroup"] = "subject-1"
        topic = ET.SubElement(subject, f"{{{mods_namespace}}}topic")
        topic.text = text
        topic.attrib["lang"] = language
