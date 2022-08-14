import pytest

@pytest.fixture(scope="session")
def default_path():
    img_file = "tests/dummy_data/test.jpg"
    xml_file = "tests/dummy_data/test.xml"

    path = {
        "img": img_file,
        "xml": xml_file
    }

    return path