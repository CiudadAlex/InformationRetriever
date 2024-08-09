"""
For processing the big XML Pubmed files we will use bigxml:
https://bigxml.rogdham.net/quickstart/
"""

from bigxml import Parser, xml_handle_element, xml_handle_text
from bigxml import BigXmlError
from bigxml import HandlerTypeHelper, Streamable, XMLElement, XMLElementAttributes, XMLText

START_ARTICLE_SYMBOL = "{{START_ARTICLE_SYMBOL}}"
END_ARTICLE_SYMBOL = "{{END_ARTICLE_SYMBOL}}"

@xml_handle_element("PubmedArticleSet", "PubmedArticle")
def handler_pubmed_article(node):
    yield START_ARTICLE_SYMBOL
    yield from node.iter_from(handler_doi, handler_article)
    yield END_ARTICLE_SYMBOL

@xml_handle_element("PubmedData", "ArticleIdList", "ArticleId")
def handler_doi(node):

    if node.attributes["IdType"] == "doi":
        yield "DOI: " + node.text  # node content as a str

@xml_handle_element("MedlineCitation", "Article")
def handler_article(node):
    yield from node.iter_from(handler_title, handler_abstract)

@xml_handle_element("ArticleTitle")
def handler_title(node):
    yield "Title: " + node.text  # node content as a str

@xml_handle_element("Abstract")
def handler_abstract(node):
    yield "Abstract: " + node.text  # node content as a str


def generate_separated_files_with_xml(file_path):

    with open(file_path, "rb") as f:
        count = 0
        list_items_doc = []
        for item in Parser(f).iter_from(handler_pubmed_article):

            if item == START_ARTICLE_SYMBOL:
                list_items_doc.clear()
            elif item == END_ARTICLE_SYMBOL:
                generate_file(list_items_doc)
                list_items_doc.clear()

                count = count + 1
                if count == 100:
                    break

            else:
                list_items_doc.append(item)

def generate_file(list_items_doc):
    print("--------------------------------------")
    print(list_items_doc)

generate_separated_files_with_xml("C:/Alex/Dev/data_corpus/InformationRetrieval/pubmed22n1513.xml")

