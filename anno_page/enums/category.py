from enum import IntEnum
from anno_page.enums.language import Language


class Category(IntEnum):
    BARCODE_AND_QR_CODE = 1
    EXLIBRIS = 2
    PHOTOGRAPH = 3
    GRAPH = 4
    CARICATURE_AND_COMICS = 5
    VIGNETTE = 6
    FRIEZE = 7
    INITIAL = 8
    SIGNET = 9
    OTHER_BOOK_DECOR = 10
    MAP = 11
    MUSICAL_NOTATION = 12
    DECORATIVE_INSCRIPTION = 13
    STAMP = 14
    ADVERTISEMENT = 15
    HANDWRITTEN_NOTE = 16
    SYMBOL_LOGO_COAT_OF_ARMS = 17
    TABLE = 18
    FLOOR_PLAN = 19
    DIAGRAM = 20
    GEOMETRIC_DRAWING = 21
    OTHER_TECHNICAL_DRAWING = 22
    MATHEMATICAL_EXPRESSION_AND_EQUATION = 23
    CHEMICAL_FORMULA_AND_EQUATION = 24
    IMAGE = 25

    @staticmethod
    def from_string(category: str):
        for language in category_to_string_mapping_reversed:
            if category in category_to_string_mapping_reversed[language]:
                return category_to_string_mapping_reversed[language][category]
        
        raise ValueError(f'Unknown category: {category}')

    def to_string(self, language: Language):
        return category_to_string_mapping[language][self]

    def __str__(self):
        return self.to_string(Language.ENGLISH)

    def to_type_of_resource(self):
        type_of_resource = "still image"

        if self == Category.MAP:
            type_of_resource = "cartographic"

        return type_of_resource


category_to_string_mapping = {
    Language.ENGLISH: {
        Category.CHEMICAL_FORMULA_AND_EQUATION: 'Chemical formula and equation',
        Category.SYMBOL_LOGO_COAT_OF_ARMS: 'Symbol, logo, coat of arms',
        Category.EXLIBRIS: 'Exlibris',
        Category.PHOTOGRAPH: 'Photograph',
        Category.GEOMETRIC_DRAWING: 'Geometric drawing',
        Category.GRAPH: 'Graph',
        Category.INITIAL: 'Initial',
        Category.CARICATURE_AND_COMICS: 'Caricature and comics',
        Category.MAP: 'Map',
        Category.MATHEMATICAL_EXPRESSION_AND_EQUATION: 'Mathematical expression and equation',
        Category.MUSICAL_NOTATION: 'Musical notation',
        Category.IMAGE: 'Image',
        Category.OTHER_BOOK_DECOR: 'Other book decor',
        Category.OTHER_TECHNICAL_DRAWING: 'Other technical drawing',
        Category.DECORATIVE_INSCRIPTION: 'Decorative inscription',
        Category.FLOOR_PLAN: 'Floor plan',
        Category.BARCODE_AND_QR_CODE: 'Barcode and QR code',
        Category.STAMP: 'Stamp',
        Category.ADVERTISEMENT: 'Advertisement',
        Category.HANDWRITTEN_NOTE: 'Handwritten note',
        Category.DIAGRAM: 'Diagram',
        Category.SIGNET: 'Signet',
        Category.TABLE: 'Table',
        Category.VIGNETTE: 'Vignette',
        Category.FRIEZE: 'Frieze'
    },
    Language.CZECH: {
        Category.CHEMICAL_FORMULA_AND_EQUATION: 'Chemický vzorec a rovnice',
        Category.SYMBOL_LOGO_COAT_OF_ARMS: 'Symbol, logo, erb',
        Category.EXLIBRIS: 'Exlibris',
        Category.PHOTOGRAPH: 'Fotografie',
        Category.GEOMETRIC_DRAWING: 'Geometrický výkres',
        Category.GRAPH: 'Graf',
        Category.INITIAL: 'Iniciála',
        Category.CARICATURE_AND_COMICS: 'Karikatura a komiks',
        Category.MAP: 'Mapa',
        Category.MATHEMATICAL_EXPRESSION_AND_EQUATION: 'Matematický výraz a rovnice',
        Category.MUSICAL_NOTATION: 'Notový zápis',
        Category.IMAGE: 'Obrázek',
        Category.OTHER_BOOK_DECOR: 'Ostatní knižní dekor',
        Category.OTHER_TECHNICAL_DRAWING: 'Ostatní technické výkresy',
        Category.DECORATIVE_INSCRIPTION: 'Ozdobný nápis',
        Category.FLOOR_PLAN: 'Půdorys',
        Category.BARCODE_AND_QR_CODE: 'Čárový a QR kód',
        Category.STAMP: 'Razítko',
        Category.ADVERTISEMENT: 'Reklama',
        Category.HANDWRITTEN_NOTE: 'Rukopisné vpisky',
        Category.DIAGRAM: 'Schéma',
        Category.SIGNET: 'Signet',
        Category.TABLE: 'Tabulka',
        Category.VIGNETTE: 'Viněta',
        Category.FRIEZE: 'Vlys'
    },
    Language.MODS_GENRE_EN: {
        Category.CHEMICAL_FORMULA_AND_EQUATION: 'chemicalFormula',
        Category.SYMBOL_LOGO_COAT_OF_ARMS: 'symbol',
        Category.EXLIBRIS: 'exlibris',
        Category.PHOTOGRAPH: 'photograph',
        Category.GEOMETRIC_DRAWING: 'geometricDrawing',
        Category.GRAPH: 'chart',
        Category.INITIAL: 'initial',
        Category.CARICATURE_AND_COMICS: 'cartoon',
        Category.MAP: 'map',
        Category.MATHEMATICAL_EXPRESSION_AND_EQUATION: 'mathematicalFormula',
        Category.MUSICAL_NOTATION: 'sheetMusic',
        Category.IMAGE: 'image',
        Category.OTHER_BOOK_DECOR: 'otherDecoration',
        Category.OTHER_TECHNICAL_DRAWING: 'otherTechnicalDrawing',
        Category.DECORATIVE_INSCRIPTION: 'decorativeInscription',
        Category.FLOOR_PLAN: 'technicalDrawing',
        Category.BARCODE_AND_QR_CODE: 'barcode',
        Category.STAMP: 'stamp',
        Category.ADVERTISEMENT: 'advertisement',
        Category.HANDWRITTEN_NOTE: 'handwrittenInscriptions',
        Category.DIAGRAM: 'schema',
        Category.SIGNET: 'signet',
        Category.TABLE: 'table',
        Category.VIGNETTE: 'vignette',
        Category.FRIEZE: 'frieze'
    },
    Language.MODS_GENRE_CZ: {
        Category.CHEMICAL_FORMULA_AND_EQUATION: 'chemickáFormule',
        Category.SYMBOL_LOGO_COAT_OF_ARMS: 'symbol',
        Category.EXLIBRIS: 'exlibris',
        Category.PHOTOGRAPH: 'fotografie',
        Category.GEOMETRIC_DRAWING: 'geometrickýVýkres',
        Category.GRAPH: 'graf',
        Category.INITIAL: 'iniciála',
        Category.CARICATURE_AND_COMICS: 'karikatura',
        Category.MAP: 'mapa',
        Category.MATHEMATICAL_EXPRESSION_AND_EQUATION: 'matematickáFormule',
        Category.MUSICAL_NOTATION: 'notovýZápis',
        Category.IMAGE: 'obrázek',
        Category.OTHER_BOOK_DECOR: 'ostatníDekorace',
        Category.OTHER_TECHNICAL_DRAWING: 'ostatníTechnickéVýkresy',
        Category.DECORATIVE_INSCRIPTION: 'ozdobnýNápis',
        Category.FLOOR_PLAN: 'technickýVýkres',
        Category.BARCODE_AND_QR_CODE: 'čárovýKód',
        Category.STAMP: 'razítko',
        Category.ADVERTISEMENT: 'reklama',
        Category.HANDWRITTEN_NOTE: 'rukopisnéVpisky',
        Category.DIAGRAM: 'schéma',
        Category.SIGNET: 'signet',
        Category.TABLE: 'tabulka',
        Category.VIGNETTE: 'viněta',
        Category.FRIEZE: 'vlys'
    }
}

category_to_string_mapping_reversed = {
    Language.ENGLISH: {v: k for k, v in category_to_string_mapping[Language.ENGLISH].items()},
    Language.CZECH: {v: k for k, v in category_to_string_mapping[Language.CZECH].items()}
}
