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

    delete_files_in_dir(output_dir_path)

    for file in os.listdir(input_dir_path):
        if file.endswith(".xml"):
            file_path = os.path.join(input_dir_path, file)
            generate_separated_files_with_xml(file_path, output_dir_path)


def delete_files_in_dir(dir_path):
    for file in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file)
        os.remove(file_path)


def generate_separated_files_with_xml(file_path, output_dir_path):

    file_name = os.path.basename(file_path)
    print("Processing file: " + file_name)
    processed_files = 0
    processed_articles = 0

    with open(file_path, "rb") as f:

        list_tuple_key_item = []
        for tuple_key_item in Parser(f).iter_from(handler_pubmed_article):

            key = tuple_key_item[0]
            item = tuple_key_item[1]

            if key == KEY_DELIMITER:

                if item == END_ARTICLE_SYMBOL:

                    processed_articles = processed_articles + 1

                    if processed_articles % 500 == 0:
                        generate_file_with_list_tuple_key_item(output_dir_path, list_tuple_key_item, file_name, processed_files)
                        processed_files = processed_files + 1

            else:
                list_tuple_key_item.append(tuple_key_item)

        generate_file_with_list_tuple_key_item(output_dir_path, list_tuple_key_item, file_name, processed_files)


def generate_file_with_list_tuple_key_item(output_dir_path, list_tuple_key_item, file_name, ident):

    output_file_path = output_dir_path + "/" + file_name + "_" + str(ident) + ".txt"
    generate_file(list_tuple_key_item, output_file_path)
    list_tuple_key_item.clear()
    print("Processed subfiles: " + str(ident))


def generate_file(list_tuple_key_item, file_path):

    content = ""
    for tuple_key_item in list_tuple_key_item:
        content = content + tuple_key_item[0] + ": " + tuple_key_item[1] + "\n"

    f = open(file_path, "w", encoding="utf-8")
    f.write(content)
    f.close()
