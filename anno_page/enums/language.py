from enum import Enum


class Language(Enum):
    ENGLISH = 0
    CZECH = 1
    MODS_GENRE_EN = 10000000
    MODS_GENRE_CZ = 10000001

    @staticmethod
    def from_string(language: str):
        return language_to_string_mapping_reversed[language]

    def to_string(self):
        return language_to_string_mapping[self]

    def __str__(self):
        return self.to_string()


language_to_string_mapping = {
    Language.ENGLISH: 'eng',
    Language.CZECH: 'cze'
}

language_to_string_mapping_reversed = {v: k for k, v in language_to_string_mapping.items()}
