import json
import uuid
import datetime
import xmltodict


def generate_uuid(*args, **kwargs):
    return uuid.UUID("01234567-0123-0123-0123-0123456789ab")


def get_datetime_now(*args, **kwargs):
    return datetime.datetime(1970, 1, 1, 12, 0, 0)


def assert_lists_equal(list1, list2):
    if len(list1) != len(list2):
        raise AssertionError(f"Lists differ in length: {len(list1)} != {len(list2)}")

    unmatched_items = list(list2)
    for item1 in list1:
        for index, item2 in enumerate(unmatched_items):
            try:
                if isinstance(item1, dict) and isinstance(item2, dict):
                    assert_dicts_equal(item1, item2)
                elif isinstance(item1, list) and isinstance(item2, list):
                    assert_lists_equal(item1, item2)
                elif item1 != item2:
                    raise AssertionError
            except AssertionError:
                continue

            del unmatched_items[index]
            break
        else:
            raise AssertionError(f"List item {item1} is missing in the second list.")


def assert_dicts_equal(dict1, dict2, load_custom_as_json=False):
    all_keys = set(dict1.keys()) | set(dict2.keys())
    for key in all_keys:
        if key not in dict1:
            raise AssertionError(f"Key '{key}' is missing in the first dictionary.")
        if key not in dict2:
            raise AssertionError(f"Key '{key}' is missing in the second dictionary.")

        value1 = dict1[key]
        value2 = dict2[key]

        # If the key is "@custom" and both values are strings that look like JSON, try to parse them as JSON
        if (load_custom_as_json and key == "@custom" and value1 != value2
                and isinstance(value1, str) and isinstance(value2, str)
                and value1.startswith("{") and value2.startswith("{")
                and value1.endswith("}") and value2.endswith("}")):
            try:
                value1 = json.loads(value1)
                value2 = json.loads(value2)
            except:
                pass

        if isinstance(value1, dict) and isinstance(value2, dict):
            assert_dicts_equal(value1, value2, load_custom_as_json=load_custom_as_json)

        elif isinstance(value1, list) and isinstance(value2, list):
            assert_lists_equal(value1, value2)

        elif value1 != value2:
            raise AssertionError(f"Values for key '{key}' differ: {value1} != {value2}")



def assert_xml_equal(actual_xml: str, expected_xml: str, load_custom_as_json: bool = False) -> None:
    actual_dict = xmltodict.parse(actual_xml)
    expected_dict = xmltodict.parse(expected_xml)
    assert_dicts_equal(actual_dict, expected_dict, load_custom_as_json=load_custom_as_json)


def load_xml(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()
