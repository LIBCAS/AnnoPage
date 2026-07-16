import logging
from typing import Optional, Dict, List
from lxml import etree as ET

from pero_ocr.core.layout import TextLine

from anno_page import globals
from anno_page.enums import Category, Language, LineRelation
from anno_page.enums.language import language_to_string_mapping, language_to_string_mapping_reversed
from anno_page.core.services import UuidService, DateTimeService

logger = logging.getLogger(__name__)


class BaseMetadata:
    def __init__(self,
                 tag_id,
                 mods_id,
                 mods_uuid=None,
                 record_identifier=None,
                 used_ai_models=None,
                 creation_date_time=None,
                 confidence=None):
        self.tag_id = tag_id
        self.mods_id = mods_id
        self.mods_uuid = str(UuidService.generate_uuid()) if mods_uuid is None else mods_uuid
        self.record_identifier = str(UuidService.generate_uuid()) if record_identifier is None else record_identifier
        self.used_ai_models = used_ai_models if used_ai_models is not None else {}
        self.creation_date_time = creation_date_time
        self.confidence = confidence

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
    def _add_record_info_element(mods, mods_namespace, creation_date_time, record_identifier, confidence=None, used_ai_models=None):
        record_info = ET.SubElement(mods, f"{{{mods_namespace}}}recordInfo")

        record_creation_date = ET.SubElement(record_info, f"{{{mods_namespace}}}recordCreationDate")
        record_creation_date.text = creation_date_time

        record_content_source = ET.SubElement(record_info, f"{{{mods_namespace}}}recordContentSource")
        record_content_source.text = globals.software_fullname

        description_standard = ET.SubElement(record_info, f"{{{mods_namespace}}}descriptionStandard")
        description_standard.text = "StandardNDK"

        record_origin = ET.SubElement(record_info, f"{{{mods_namespace}}}recordOrigin")
        record_origin.text = "machine-generated"

        if confidence is not None:
            record_info_note = ET.SubElement(record_info, f"{{{mods_namespace}}}recordInfoNote")
            record_info_note.attrib["type"] = "confidence"
            record_info_note.text = f"{confidence:.3f}"

        if used_ai_models is not None:
            for model_type, model_name in used_ai_models.items():
                record_info_note = ET.SubElement(record_info, f"{{{mods_namespace}}}recordInfoNote")
                record_info_note.attrib["type"] = str(model_type)
                record_info_note.text = model_name

        record_identifier_element = ET.SubElement(record_info, f"{{{mods_namespace}}}recordIdentifier")
        record_identifier_element.attrib["source"] = globals.software_name
        record_identifier_element.text = f"uuid:{record_identifier}"

        language_of_cataloging = ET.SubElement(record_info, f"{{{mods_namespace}}}languageOfCataloging")

        language_term_en = ET.SubElement(language_of_cataloging, f"{{{mods_namespace}}}languageTerm")
        language_term_en.text = str(Language.ENGLISH)
        language_term_en.attrib["authority"] = "iso639-2b"

        language_term_cz = ET.SubElement(language_of_cataloging, f"{{{mods_namespace}}}languageTerm")
        language_term_cz.text = str(Language.CZECH)
        language_term_cz.attrib["authority"] = "iso639-2b"

    def to_altoxml(self, *args, **kwargs):
        raise NotImplementedError

    def to_dict(self) -> dict:
        return {
            "tag_id": self.tag_id,
            "mods_id": self.mods_id,
            "mods_uuid": str(self.mods_uuid),
            "record_identifier": str(self.record_identifier),
            "used_ai_models": self.used_ai_models,
            "creation_date_time": self.creation_date_time,
            "confidence": self.confidence
        }


class RelatedLinesMetadata(BaseMetadata):
    def __init__(self,
                 tag_id,
                 mods_id,
                 lines: List[TextLine],
                 relation: LineRelation,
                 description: Optional[str] = None,
                 title: Optional[str | Dict[Language, str]] = None,
                 mods_uuid=None,
                 record_identifier=None,
                 creation_date_time=None,
                 used_ai_models: Optional[Dict[str, str]] = None,
                 confidence: Optional[float] = None):
        super().__init__(tag_id=tag_id,
                         mods_id=mods_id,
                         mods_uuid=mods_uuid,
                         record_identifier=record_identifier,
                         creation_date_time=creation_date_time,
                         used_ai_models=used_ai_models,
                         confidence=confidence)
        self.lines = lines
        self.relation = relation
        self.description = description
        self.title = title

    def to_altoxml(self, tags, mods_namespace, related_mods_id=None):
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
                "genre": "figureCaption",
                "genre_en": "caption",
                "genre_cz": "popisek",
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

        if self.creation_date_time is not None:
            creation_date_time = self.creation_date_time
        else:
            creation_date_time = DateTimeService.get_datetime_now().isoformat(timespec='seconds')

        self._add_record_info_element(mods,
                                      mods_namespace,
                                      creation_date_time=creation_date_time,
                                      record_identifier=self.record_identifier,
                                      confidence=self.confidence,
                                      used_ai_models=self.used_ai_models)

    @classmethod
    def from_altoxml(cls, page_layout, tag_element, tags_element, print_space_element, relation_type):
        tag_id = tag_element.attrib["ID"]
        if tag_id is None:
            return None

        mods_data = tag_element.find(f".//XmlData/mods:mods", namespaces=tags_element.nsmap)
        if mods_data is None:
            return None

        mods_id = mods_data.attrib["ID"]
        if mods_id is None:
            return None

        description = tag_element.attrib.get("DESCRIPTION", None)

        lines = cls.find_related_lines(page_layout, tag_id, print_space_element)
        result = cls(tag_id=tag_id, mods_id=mods_id, description=description, lines=lines, relation=relation_type)

        result = result.from_altoxml_mods(mods_data)

        return result

    @staticmethod
    def find_related_lines(page_layout, tag_id, print_space_element):
        lines = []
        lines_id = set()

        text_line_elements = print_space_element.findall(f".//TextLine", namespaces=print_space_element.nsmap)
        for text_line_element in text_line_elements:
            tagrefs = text_line_element.attrib.get("TAGREFS", None)
            if tagrefs is not None:
                tagref_list = tagrefs.split()
                if tag_id in tagref_list:
                    lines_id.add(text_line_element.attrib["ID"])

        for line in page_layout.lines_iterator():
            if line.id in lines_id:
                lines.append(line)

        return lines

    def from_altoxml_mods(self, mods_data):
        title = self.from_altoxml_mods_title(mods_data)
        mods_uuid = self.from_altoxml_mods_uuid(mods_data)
        record_identifier = self.from_altoxml_mods_record_identifier(mods_data)
        creation_date_time = self.from_altoxml_mods_creation_date_time(mods_data)
        confidence = self.from_altoxml_mods_confidence(mods_data)

        self.title = title
        self.mods_uuid = mods_uuid
        self.record_identifier = record_identifier
        self.creation_date_time = creation_date_time
        self.confidence = confidence

        return self

    @staticmethod
    def from_altoxml_mods_title(mods_data):
        title_info_elements = mods_data.findall("mods:titleInfo", mods_data.nsmap)
        if not title_info_elements:
            return None

        titles = {}
        for title_info in title_info_elements:
            title_element = title_info.find("mods:title", mods_data.nsmap)
            if title_element is not None:
                lang = title_element.attrib.get("lang", None)
                text = title_element.text

                if text:
                    text = text.strip()

                if lang is not None:
                    lang_enum = language_to_string_mapping_reversed.get(lang, None)
                    if lang_enum is not None:
                        titles[lang_enum] = text
                else:
                    titles[None] = text

        if len(titles) == 1 and None in titles:
            return titles[None]
        elif len(titles) > 0:
            return titles
        else:
            return None

    @staticmethod
    def from_altoxml_mods_description(mods_data):
        description_elements = mods_data.findall("mods:abstract[@type='description']", mods_data.nsmap)
        if not description_elements:
            return None

        descriptions = {}
        for desc in description_elements:
            lang = desc.attrib.get("lang", None)
            text = desc.text

            if text:
                text = text.strip()

            if lang is not None:
                lang_enum = language_to_string_mapping_reversed.get(lang, None)
                if lang_enum is not None:
                    descriptions[lang_enum] = text
            else:
                descriptions[None] = text

        if len(descriptions) == 1 and None in descriptions:
            return descriptions[None]
        elif len(descriptions) > 0:
            return descriptions
        else:
            return None

    @staticmethod
    def from_altoxml_mods_uuid(mods_data):
        identifier_element = mods_data.find("mods:identifier[@type='uuid']", mods_data.nsmap)
        if identifier_element is not None:
            identifier_text = identifier_element.text

            if identifier_text:
                identifier_text = identifier_text.strip()

            if identifier_text.startswith("uuid:"):
                return identifier_text[len("uuid:"):]
            else:
                return identifier_text
        else:
            return None

    @staticmethod
    def from_altoxml_mods_record_identifier(mods_data):
        record_identifier_element = mods_data.find("mods:recordInfo/mods:recordIdentifier", mods_data.nsmap)
        if record_identifier_element is not None:
            record_identifier_text = record_identifier_element.text

            if record_identifier_text:
                record_identifier_text = record_identifier_text.strip()

            if record_identifier_text.startswith("uuid:"):
                return record_identifier_text[len("uuid:"):]
            else:
                return record_identifier_text
        else:
            return None


    @staticmethod
    def from_altoxml_mods_creation_date_time(mods_data):
        record_info_date_elements = mods_data.findall("mods:recordInfo/mods:recordCreationDate", mods_data.nsmap)
        if not record_info_date_elements:
            return None

        for date_element in record_info_date_elements:
            text = date_element.text

            if text:
                text = text.strip()
                return text

        return None

    @staticmethod
    def from_altoxml_mods_confidence(mods_data):
        record_info_note_elements = mods_data.findall("mods:recordInfo/mods:recordInfoNote[@type='confidence']", mods_data.nsmap)

        confidence = None
        for note in record_info_note_elements:
            if note.text:
                try:
                    confidence = float(note.text.strip())
                except ValueError:
                    continue

        return confidence

    def to_dict(self) -> dict:
        result = super().to_dict()

        title = {language_to_string_mapping[k]: v for k, v in self.title.items()} if isinstance(self.title, dict) else self.title

        result.update({
            "lines": [line.id for line in self.lines],
            "relation": self.relation.value,
            "description": self.description,
            "title": title
        })

        return result

    def update(self, other):
        if other is None:
            return

        description = self.description if self.description is not None else ""
        title = self.title if self.title is not None else ""

        existing_line_ids = {line.id for line in self.lines}
        for line in other.lines:
            if line.id not in existing_line_ids:
                self.lines.append(line)

        description += f" {other.description}" if other.description is not None else ""
        title += f" {other.title}" if other.title is not None else ""

        self.description = description.strip() if description else None
        self.title = title.strip() if title else None


class GraphicalObjectMetadata(BaseMetadata):
    def __init__(self, 
                 tag_id,
                 mods_id,
                 mods_uuid=None,
                 record_identifier=None,
                 creation_date_time=None,
                 tag_description: Optional[str] = None,
                 description: Optional[str| Dict[Language, str]] = None,
                 caption: Optional[str | Dict[Language, str]] = None,
                 topics: Optional[str | Dict[Language, str] | Dict[Language, list[str]]] = None,
                 color: Optional[str | Dict[Language, str]] = None,
                 title: Optional[str | Dict[Language, str]] = None,
                 caption_lines_metadata: Optional[RelatedLinesMetadata] = None,
                 reference_lines_metadata: Optional[RelatedLinesMetadata] = None,
                 continuing_line: Optional[TextLine] = None,
                 prompts: Optional[List[str]] = None,
                 used_ai_models: Optional[Dict[str, str]] = None,
                 confidence: Optional[float] = None):
        super().__init__(tag_id=tag_id,
                         mods_id=mods_id,
                         mods_uuid=mods_uuid,
                         record_identifier=record_identifier,
                         creation_date_time=creation_date_time,
                         used_ai_models=used_ai_models,
                         confidence=confidence)
        self.tag_description = tag_description
        self.caption = caption
        self.topics = topics
        self.color = color
        self.description = description
        self.title = title
        self.continuing_line = continuing_line
        self.prompts = prompts

        self.caption_lines_metadata = caption_lines_metadata
        self.reference_lines_metadata = reference_lines_metadata

    def update(self, other, merge_caption_lines=True, merge_reference_lines=True):
        if other is None:
            return

        if self.caption is None and other.caption is not None:
            self.caption = other.caption
        if self.topics is None and other.topics is not None:
            self.topics = other.topics
        if self.color is None and other.color is not None:
            self.color = other.color
        if self.description is None and other.description is not None:
            self.description = other.description
        if self.title is None and other.title is not None:
            self.title = other.title

        if self.caption_lines_metadata is None:
            self.caption_lines_metadata = other.caption_lines_metadata
        else:
            if merge_caption_lines:
                self.caption_lines_metadata.update(other.caption_lines_metadata)
            else:
                self.caption_lines_metadata = other.caption_lines_metadata

        if self.reference_lines_metadata is None:
            self.reference_lines_metadata = other.reference_lines_metadata
        else:
            if merge_reference_lines:
                self.reference_lines_metadata.update(other.reference_lines_metadata)
            else:
                self.reference_lines_metadata = other.reference_lines_metadata

    def to_altoxml(self, tags, mods_namespace, category, bounding_box):
        self.graphics_to_altoxml(tags, mods_namespace, category, bounding_box)

        if self.caption_lines_metadata is not None:
            self.caption_lines_metadata.to_altoxml(tags, mods_namespace, self.mods_id)

        if self.reference_lines_metadata is not None:
            self.reference_lines_metadata.to_altoxml(tags, mods_namespace, self.mods_id)

    def graphics_to_altoxml(self, tags, mods_namespace, category, bounding_box):
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

        if self.tag_description is not None:
            layout_tag.set("DESCRIPTION", self.tag_description)

        if self.title is not None:
            self._add_title_element(mods, mods_namespace, title=self.title)
        elif self.caption_lines_metadata is not None and self.caption_lines_metadata.title is not None:
            self._add_title_element(mods, mods_namespace, title=self.caption_lines_metadata.title)

        self._add_type_of_resource_element(mods, mods_namespace, category.to_type_of_resource())

        self._add_genre_element(mods, mods_namespace, str(Language.ENGLISH), category_mods_en)
        self._add_genre_element(mods, mods_namespace, str(Language.CZECH), category_mods_cz)

        self._add_size_element(mods, mods_namespace, bounding_box)

        if self.color is not None:
            self._add_color_elements(mods, mods_namespace)

        if self.caption is not None:
            self._add_caption_elements(mods, mods_namespace)

        if self.description is not None:
            self._add_description_elements(mods, mods_namespace)

        if self.topics is not None:
            self._add_topics_elements(mods, mods_namespace)

        self._add_identifier_element(mods, mods_namespace, self.mods_uuid, identifier_type="uuid")

        if self.caption_lines_metadata is not None:
            self._add_related_item_element(mods, mods_namespace, "constituent", self.caption_lines_metadata.mods_id)

        if self.reference_lines_metadata is not None:
            self._add_related_item_element(mods, mods_namespace, "references", self.reference_lines_metadata.mods_id)

        if self.creation_date_time is not None:
            creation_date_time = self.creation_date_time
        else:
            creation_date_time = DateTimeService.get_datetime_now().isoformat(timespec='seconds')

        self._add_record_info_element(mods,
                                      mods_namespace,
                                      creation_date_time=creation_date_time,
                                      record_identifier=self.record_identifier,
                                      confidence=self.confidence,
                                      used_ai_models=self.used_ai_models)

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
                if color is not None:
                    self._add_color_element(mods, mods_namespace, str(language), color)
        else:
            logger.warning(f"Color is not a string or dictionary in GraphicalObjectMetadata '{self.tag_id}'.")

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
                if caption is not None:
                    self._add_caption_element(mods, mods_namespace, str(language), caption)
        else:
            logger.warning(f"Caption is not a string or dictionary in GraphicalObjectMetadata '{self.tag_id}'.")

    @staticmethod
    def _add_caption_element(mods, mods_namespace, language, text):
        abstract = ET.SubElement(mods, f"{{{mods_namespace}}}abstract")
        abstract.text = text
        abstract.attrib["altRepGroup"] = "caption-1"
        abstract.attrib["type"] = "caption"
        abstract.attrib["lang"] = language

    def _add_topics_elements(self, mods, mods_namespace):
        if isinstance(self.topics, str) or isinstance(self.topics, list):
            self._add_topic_elements(mods, mods_namespace, "", self.topics)
        elif isinstance(self.topics, dict):
            for language, topic in self.topics.items():
                if topic is not None:
                    self._add_topic_elements(mods, mods_namespace, str(language), topic)
        else:
            logger.warning(f"Topic is not a string or dictionary in GraphicalObjectMetadata '{self.tag_id}'.")

    def _add_topic_elements(self, mods, mods_namespace, language, topics):
        if isinstance(topics, list):
            topic_list = topics
        elif isinstance(topics, str):
            topic_list = topics.split(",")
        else:
            logger.warning(f"Topics is not a string or list in GraphicalObjectMetadata '{self.tag_id}'.")
            return

        for index, topic_text in enumerate(topic_list):
            self._add_topic_element(mods, mods_namespace, language, topic_text.strip(), index + 1)

    @staticmethod
    def _add_topic_element(mods, mods_namespace, language, text, index):
        subject = ET.SubElement(mods, f"{{{mods_namespace}}}subject")
        subject.attrib["altRepGroup"] = f"subject-{index}"
        topic = ET.SubElement(subject, f"{{{mods_namespace}}}topic")
        topic.text = text
        topic.attrib["lang"] = language

    def _add_description_elements(self, mods, mods_namespace):
        if isinstance(self.description, str):
            self._add_description_element(mods, mods_namespace, None, self.description)
        elif isinstance(self.description, dict):
            for language, description in self.description.items():
                if description is not None:
                    self._add_description_element(mods, mods_namespace, str(language), description)
        else:
            logger.warning(f"Description is not a string or dictionary in GraphicalObjectMetadata '{self.tag_id}'.")

    @staticmethod
    def _add_description_element(mods, mods_namespace, language, text):
        abstract = ET.SubElement(mods, f"{{{mods_namespace}}}abstract")
        abstract.text = text
        abstract.attrib["altRepGroup"] = "description-1"
        abstract.attrib["type"] = "description"

        if language is not None:
            abstract.attrib["lang"] = language

    def to_dict(self) -> dict:
        result = super().to_dict()

        description = {language_to_string_mapping[k]: v for k, v in self.description.items()} if isinstance(self.description, dict) else self.description
        caption = {language_to_string_mapping[k]: v for k, v in self.caption.items()} if isinstance(self.caption, dict) else self.caption
        topics = {language_to_string_mapping[k]: v for k, v in self.topics.items()} if isinstance(self.topics, dict) else self.topics
        color = {language_to_string_mapping[k]: v for k, v in self.color.items()} if isinstance(self.color, dict) else self.color
        title = {language_to_string_mapping[k]: v for k, v in self.title.items()} if isinstance(self.title, dict) else self.title

        result.update({
            "description": description,
            "caption": caption,
            "topics": topics,
            "color": color,
            "title": title,
            "tag_description": self.tag_description,
            "caption_lines_metadata": self.caption_lines_metadata.to_dict() if self.caption_lines_metadata is not None else None,
            "reference_lines_metadata": self.reference_lines_metadata.to_dict() if self.reference_lines_metadata is not None else None
        })

        return result

    @classmethod
    def from_altoxml(cls, page_layout, tag_element, tags_element, print_space_element):
        tag_type = tag_element.attrib["TYPE"]
        if tag_type != "Structural":
            return None

        tag_id = tag_element.attrib["ID"]
        if tag_id is None:
            return None

        tag_description = tag_element.attrib.get("DESCRIPTION", None)

        mods_data = tag_element.find(f".//XmlData/mods:mods", namespaces=tags_element.nsmap)
        if mods_data is None:
            return None

        mods_id = mods_data.attrib["ID"]
        if mods_id is None:
            return None

        result = cls(tag_id=tag_id, mods_id=mods_id, tag_description=tag_description)
        result = result.from_altoxml_mods(page_layout, mods_data, tags_element)

        caption_lines_mods_tag_id, reference_lines_mods_tag_id = cls.find_related_lines_mods_tags_id(tags_element, mods_data)

        ns = {
            "alto": tags_element.nsmap[None],
            "mods": tags_element.nsmap["mods"],
        }

        caption_lines_tags = tags_element.xpath("alto:StructureTag[alto:XmlData/mods:mods[@ID=$mods_id]]", namespaces=ns, mods_id=caption_lines_mods_tag_id) if caption_lines_mods_tag_id else None
        reference_lines_tags = tags_element.xpath("alto:OtherTag[alto:XmlData/mods:mods[@ID=$mods_id]]", namespaces=ns, mods_id=reference_lines_mods_tag_id) if reference_lines_mods_tag_id else None

        caption_lines_tag = caption_lines_tags[0] if caption_lines_tags else None
        reference_lines_tag = reference_lines_tags[0] if reference_lines_tags else None

        if caption_lines_tag is not None:
            caption_lines_metadata = RelatedLinesMetadata.from_altoxml(page_layout, caption_lines_tag, tags_element, print_space_element, relation_type=LineRelation.CAPTION)
            result.caption_lines_metadata = caption_lines_metadata

        if reference_lines_tag is not None:
            reference_lines_metadata = RelatedLinesMetadata.from_altoxml(page_layout, reference_lines_tag, tags_element, print_space_element, relation_type=LineRelation.REFERENCE)
            result.reference_lines_metadata = reference_lines_metadata

        return result

    @staticmethod
    def find_related_lines_mods_tags_id(tags_element, mods_data):
        caption_lines_mods_tag_id = None
        reference_lines_mods_tag_id = None

        related_items = mods_data.findall("mods:relatedItem", mods_data.nsmap)
        for related_item in related_items:
            item_type = related_item.attrib.get("type", None)
            if item_type == "constituent":
                caption_lines_mods_tag_id = related_item.attrib["IDREF"]
            elif item_type == "references":
                reference_lines_mods_tag_id = related_item.attrib["IDREF"]

        return caption_lines_mods_tag_id, reference_lines_mods_tag_id

    def from_altoxml_mods(self, page_layout, mods_data, tags_element):
        description = self.from_altoxml_mods_description(mods_data)
        caption = self.from_altoxml_mods_caption(mods_data)
        topics = self.from_altoxml_mods_topics(mods_data)
        color = self.from_altoxml_mods_color(mods_data)
        title = self.from_altoxml_mods_title(mods_data)
        mods_uuid = self.from_altoxml_mods_uuid(mods_data)
        record_identifier = self.from_altoxml_mods_record_identifier(mods_data)
        used_ai_models = self.from_altoxml_mods_used_ai_models(mods_data)
        confidence = self.from_altoxml_mods_confidence(mods_data)
        creation_date_time = self.from_altoxml_mods_creation_date_time(mods_data)

        self.description = description
        self.caption = caption
        self.topics = topics
        self.color = color
        self.title = title
        self.mods_uuid = mods_uuid
        self.record_identifier = record_identifier
        self.used_ai_models = used_ai_models
        self.confidence = confidence
        self.creation_date_time = creation_date_time

        return self

    @staticmethod
    def from_altoxml_mods_description(mods_data):
        description_elements = mods_data.findall("mods:abstract[@type='description']", mods_data.nsmap)
        if not description_elements:
            return None

        descriptions = {}
        for desc in description_elements:
            lang = desc.attrib.get("lang", None)
            text = desc.text

            if text:
                text = text.strip()

            if lang is not None:
                lang_enum = language_to_string_mapping_reversed.get(lang, None)
                if lang_enum is not None:
                    descriptions[lang_enum] = text
            else:
                descriptions[None] = text

        if len(descriptions) == 1 and None in descriptions:
            return descriptions[None]
        elif len(descriptions) > 0:
            return descriptions
        else:
            return None

    @staticmethod
    def from_altoxml_mods_caption(mods_data):
        caption_elements = mods_data.findall("mods:abstract[@type='caption']", mods_data.nsmap)
        if not caption_elements:
            return None

        captions = {}
        for cap in caption_elements:
            lang = cap.attrib.get("lang", None)
            text = cap.text

            if text:
                text = text.strip()

            if lang is not None:
                lang_enum = language_to_string_mapping_reversed.get(lang, None)
                if lang_enum is not None:
                    captions[lang_enum] = text
            else:
                captions[None] = text

        if len(captions) == 1 and None in captions:
            return captions[None]
        elif len(captions) > 0:
            return captions
        else:
            return None

    @staticmethod
    def from_altoxml_mods_topics(mods_data):
        subject_elements = mods_data.findall("mods:subject", mods_data.nsmap)
        if not subject_elements:
            return None

        topics = {}
        for subj in subject_elements:
            topic_element = subj.find("mods:topic", mods_data.nsmap)
            if topic_element is not None:
                lang = topic_element.attrib.get("lang", None)
                text = topic_element.text

                if text:
                    text = text.strip()

                if lang is not None:
                    lang_enum = language_to_string_mapping_reversed.get(lang, None)
                    if lang_enum is not None:
                        if lang_enum not in topics:
                            topics[lang_enum] = []
                        topics[lang_enum].append(text)
                else:
                    if None not in topics:
                        topics[None] = []
                    topics[None].append(text)

        for lang in topics:
            if len(topics[lang]) == 1 and None in topics:
                topics[lang] = topics[lang][0]

        return topics if len(topics) > 0 else None

    @staticmethod
    def from_altoxml_mods_color(mods_data):
        form_elements = mods_data.findall("mods:physicalDescription/mods:form[@type='color']", mods_data.nsmap)
        if not form_elements:
            return None

        colors = {}
        for form in form_elements:
            lang = form.attrib.get("lang", None)
            text = form.text

            if text:
                text = text.strip()

            if lang is not None:
                lang_enum = language_to_string_mapping_reversed.get(lang, None)
                if lang_enum is not None:
                    colors[lang_enum] = text
            else:
                colors[None] = text

        if len(colors) == 1 and None in colors:
            return colors[None]
        elif len(colors) > 0:
            return colors
        else:
            return None

    @staticmethod
    def from_altoxml_mods_title(mods_data):
        title_info_elements = mods_data.findall("mods:titleInfo", mods_data.nsmap)
        if not title_info_elements:
            return None

        titles = {}
        for title_info in title_info_elements:
            title_element = title_info.find("mods:title", title_info.nsmap)
            if title_element is not None:
                lang = title_element.attrib.get("lang", None)
                text = title_element.text

                if text:
                    text = text.strip()

                if lang is not None:
                    lang_enum = language_to_string_mapping_reversed.get(lang, None)
                    if lang_enum is not None:
                        titles[lang_enum] = text
                else:
                    titles[None] = text

        if len(titles) == 1 and None in titles:
            return titles[None]
        elif len(titles) > 0:
            return titles
        else:
            return None

    @staticmethod
    def from_altoxml_mods_uuid(mods_data):
        identifier_element = mods_data.find("mods:identifier[@type='uuid']", mods_data.nsmap)
        if identifier_element is not None:
            text = identifier_element.text

            if text:
                text = text.strip()

            if text.startswith("uuid:"):
                return text[len("uuid:"):]
            else:
                return text
        return None

    @staticmethod
    def from_altoxml_mods_record_identifier(mods_data):
        record_identifier_element = mods_data.find("mods:recordInfo/mods:recordIdentifier", mods_data.nsmap)
        if record_identifier_element is not None:
            text = record_identifier_element.text

            if text:
                text = text.strip()

            if text.startswith("uuid:"):
                return text[len("uuid:"):]
            else:
                return text
        return None

    @staticmethod
    def from_altoxml_mods_used_ai_models(mods_data):
        record_info_note_elements = mods_data.findall("mods:recordInfo/mods:recordInfoNote", mods_data.nsmap)
        record_info_note_elements = [note for note in record_info_note_elements if note.attrib.get("type", None) != "confidence"]
        if not record_info_note_elements:
            return None

        used_ai_models = {}
        for note in record_info_note_elements:
            model_type = note.attrib.get("type", None)
            text = note.text

            if text:
                text = text.strip()

            if model_type is not None and text is not None:
                used_ai_models[model_type] = text

        return used_ai_models if len(used_ai_models) > 0 else None

    @staticmethod
    def from_altoxml_mods_confidence(mods_data):
        record_info_note_elements = mods_data.findall("mods:recordInfo/mods:recordInfoNote[@type='confidence']", mods_data.nsmap)

        confidence = None
        for note in record_info_note_elements:
            if note.text:
                try:
                    confidence = float(note.text.strip())
                except ValueError:
                    continue

        return confidence

    @staticmethod
    def from_altoxml_mods_creation_date_time(mods_data):
        record_info_date_elements = mods_data.findall("mods:recordInfo/mods:recordCreationDate", mods_data.nsmap)
        if not record_info_date_elements:
            return None

        for date_element in record_info_date_elements:
            text = date_element.text

            if text:
                text = text.strip()
                return text

        return None

