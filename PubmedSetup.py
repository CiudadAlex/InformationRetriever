"""
For processing the big XML Pubmed files we will use bigxml:
https://bigxml.rogdham.net/quickstart/
"""

from bigxml import Parser, xml_handle_element
import os

START_ARTICLE_SYMBOL = "{{START_ARTICLE_SYMBOL}}"
END_ARTICLE_SYMBOL = "{{END_ARTICLE_SYMBOL}}"

KEY_DELIMITER = "DELIMITER"
KEY_DOI = "DOI"
KEY_TITLE = "Title"
KEY_ABSTRACT = "Abstract"


@xml_handle_element("PubmedArticleSet", "PubmedArticle")
def handler_pubmed_article(node):
    yield KEY_DELIMITER, START_ARTICLE_SYMBOL
    yield from node.iter_from(handler_doi, handler_article)
    yield KEY_DELIMITER, END_ARTICLE_SYMBOL


@xml_handle_element("PubmedData", "ArticleIdList", "ArticleId")
def handler_doi(node):
    if node.attributes["IdType"] == "doi":
        yield KEY_DOI, node.text


@xml_handle_element("MedlineCitation", "Article")
def handler_article(node):
    yield from node.iter_from(handler_title, handler_abstract)


@xml_handle_element("ArticleTitle")
def handler_title(node):
    yield KEY_TITLE, node.text


@xml_handle_element("Abstract")
def handler_abstract(node):
    yield KEY_ABSTRACT, node.text  # node content as a str


def generate_separated_files_of_xml_in_dir(input_dir_path, output_dir_path):
    for file in os.listdir(input_dir_path):
        if file.endswith(".xml"):
            file_path = os.path.join(input_dir_path, file)
            generate_separated_files_with_xml(file_path, output_dir_path)


def generate_separated_files_with_xml(file_path, output_dir_path):
    print("Processing file: " + file_path)

    with open(file_path, "rb") as f:
        count = 0
        list_items_doc = []
        for tuple_key_item in Parser(f).iter_from(handler_pubmed_article):

            key = tuple_key_item[0]
            item = tuple_key_item[1]

            if key == KEY_DELIMITER:

                if item == START_ARTICLE_SYMBOL:
                    list_items_doc.clear()
                elif item == END_ARTICLE_SYMBOL:
                    generate_file(list_items_doc)
                    list_items_doc.clear()

                    count = count + 1
                    if count == 100:
                        break

            else:
                list_items_doc.append(tuple_key_item)


def generate_file(list_items_doc):
    print("--------------------------------------")
    print(list_items_doc)
